"""Run a bounded, CPU-only gate/up format preflight using verified ggml DLLs.

No model weights are read or changed. This is synthetic routed-operation timing,
not a whole-model throughput or quality forecast. Run only when model workers
are idle. Compile the small helper first; no production engine rebuild occurs.
"""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import statistics
import shutil
import subprocess
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
FOLDER = ROOT / "data/optimization-final-gateup/kernels"
BIN = ROOT / "third_party/llama.cpp/build-ninja-release/bin"
sys.path.insert(0, str(ROOT / "third_party/llama.cpp/gguf-py"))
import gguf


def digest(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            sha.update(block)
    return sha.hexdigest()


def write(path: Path, data) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def stage_runtime(runtime_bin: Path, runtime_source: Path, exe: Path, folder: Path):
    """Verify helper ABI, stage exact DLLs and inspect exports without running a kernel."""
    private_source = ROOT / "third_party/llama.cpp"
    header_receipts = {}
    for relative in ("ggml/include/ggml.h", "ggml/include/ggml-backend.h",
                     "ggml/include/ggml-cpu.h", "ggml/include/ggml-alloc.h"):
        compiled_header = private_source / relative
        runtime_header = runtime_source / relative
        expected = digest(compiled_header)
        actual = digest(runtime_header)
        if actual != expected:
            raise RuntimeError("Runtime header differs from helper ABI; compile a separate helper against the production source: " + relative)
        header_receipts[relative] = {"helper_header_sha256": expected, "runtime_header_sha256": actual, "identical": True}
    staged = folder / "runtime"
    staged.mkdir(parents=True, exist_ok=True)
    receipts = {}
    for name in ("ggml-base.dll", "ggml-cpu.dll"):
        source = runtime_bin / name
        target = staged / name
        fingerprint = digest(source)
        if target.exists() and digest(target) != fingerprint:
            raise RuntimeError("Staged runtime differs; preserve it and choose a fresh output directory")
        if not target.exists():
            shutil.copy2(source, target)
        if digest(target) != fingerprint:
            raise RuntimeError("DLL copy verification failed: " + name)
        receipts[name] = {"source": str(source), "staged": str(target), "bytes": target.stat().st_size, "sha256": fingerprint}
    staged_exe = staged / exe.name
    executable_fingerprint = digest(exe)
    if staged_exe.exists() and digest(staged_exe) != executable_fingerprint:
        raise RuntimeError("Staged helper differs; choose a fresh output directory")
    if not staged_exe.exists():
        shutil.copy2(exe, staged_exe)
    assert digest(staged_exe) == executable_fingerprint

    class TypeTraits(ctypes.Structure):
        _fields_ = [("type_name", ctypes.c_char_p), ("blck_size", ctypes.c_int64),
                    ("blck_size_interleave", ctypes.c_int64), ("type_size", ctypes.c_size_t),
                    ("is_quantized", ctypes.c_bool), ("to_float", ctypes.c_void_p),
                    ("from_float_ref", ctypes.c_void_p)]

    directory = os.add_dll_directory(str(staged))
    base = ctypes.CDLL(str(staged / "ggml-base.dll"))
    cpu = ctypes.CDLL(str(staged / "ggml-cpu.dll"))
    exports = {"ggml-base.dll": ("ggml_init", "ggml_tensor_overhead", "ggml_graph_overhead",
        "ggml_new_tensor_3d", "ggml_new_tensor_2d", "ggml_mul_mat_id", "ggml_row_size", "ggml_nbytes",
        "ggml_backend_get_device", "ggml_backend_get_default_buffer_type", "ggml_backend_reg_get_proc_address",
        "ggml_backend_dev_supports_op", "ggml_backend_buft_alloc_buffer", "ggml_backend_buffer_free",
        "ggml_backend_alloc_ctx_tensors_from_buft", "ggml_backend_alloc_ctx_tensors", "ggml_backend_buffer_clear",
        "ggml_backend_tensor_set", "ggml_backend_tensor_get", "ggml_backend_graph_compute", "ggml_backend_free",
        "ggml_get_type_traits", "ggml_quantize_chunk", "ggml_quantize_free", "ggml_new_graph",
        "ggml_build_forward_expand", "ggml_gallocr_new", "ggml_gallocr_alloc_graph", "ggml_gallocr_free", "ggml_free"),
        "ggml-cpu.dll": ("ggml_backend_cpu_init", "ggml_backend_cpu_set_n_threads", "ggml_backend_cpu_reg")}
    for name, library in (("ggml-base.dll", base), ("ggml-cpu.dll", cpu)):
        for symbol in exports[name]:
            if not hasattr(library, symbol):
                raise RuntimeError("Runtime lacks helper export: " + name + "/" + symbol)
    getter = base.ggml_get_type_traits
    getter.argtypes = [ctypes.c_int]
    getter.restype = ctypes.POINTER(TypeTraits)
    formats = {}
    for name in ("IQ2_XXS", "IQ2_XS", "IQ2_S", "IQ3_S", "IQ3_XXS", "Q2_0"):
        kind = getattr(gguf.GGMLQuantizationType, name)
        traits = getter(int(kind)).contents
        expected_block, expected_bytes = gguf.GGML_QUANT_SIZES[kind]
        if traits.type_name.decode().lower() != name.lower() or traits.blck_size != expected_block or traits.type_size != expected_bytes or not traits.is_quantized or not traits.to_float:
            raise RuntimeError("Runtime type table mismatches verified GGUF ABI: " + name)
        formats[name] = {"enum_id": int(kind), "name": traits.type_name.decode(), "block_values": traits.blck_size,
                         "block_bytes": traits.type_size, "to_float_export_present": bool(traits.to_float)}
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    module_path = kernel.GetModuleFileNameW
    module_path.argtypes = [ctypes.c_void_p, ctypes.c_wchar_p, ctypes.c_uint32]
    module_path.restype = ctypes.c_uint32
    loaded = {}
    for name, library in (("ggml-base.dll", base), ("ggml-cpu.dll", cpu)):
        buffer = ctypes.create_unicode_buffer(32768)
        if not module_path(library._handle, buffer, len(buffer)):
            raise ctypes.WinError(ctypes.get_last_error())
        actual_path = Path(buffer.value).resolve()
        if actual_path != (staged / name).resolve():
            raise RuntimeError("Unexpected DLL loaded during ABI preflight: " + str(actual_path))
        loaded[name] = str(actual_path)
    directory.close()
    revision = subprocess.run(["git", "-C", str(runtime_source), "rev-parse", "HEAD"],
        capture_output=True, text=True, timeout=15, check=True).stdout.strip()
    receipt = {"status": "ready", "runtime_bin": str(runtime_bin), "runtime_source": str(runtime_source),
        "runtime_revision": revision, "staged_runtime": str(staged), "staged_helper": str(staged_exe),
        "dll_receipts": receipts, "helper_sha256": executable_fingerprint, "header_abi": header_receipts,
        "exports_present": {name: list(symbols) for name, symbols in exports.items()}, "format_traits": formats,
        "loaded_module_paths": loaded, "checked_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Load exact CPU/base DLLs, inspect exports and constant type tables; no CPU backend, kernel, model or GPU workload launched"}
    write(folder / "runtime-preflight.json", receipt)
    return staged_exe, staged, receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--timeout", type=int, default=240, help="Maximum seconds for each format/layout process")
    parser.add_argument("--formats", nargs="+", default=["IQ2_XXS", "IQ2_XS", "IQ2_S", "IQ3_S", "IQ3_XXS"])
    parser.add_argument("--layouts", nargs="+", choices=["native", "preferred"], default=["native", "preferred"])
    parser.add_argument("--executable", type=Path, default=FOLDER / "gateup-format-bench.exe")
    parser.add_argument("--runtime-bin", type=Path, help="CPU/base DLL source; staged into an isolated directory under --output")
    parser.add_argument("--runtime-source", type=Path, help="Matching runtime checkout; its API headers must equal the helper headers")
    parser.add_argument("--output", type=Path, default=FOLDER, help="Separate output directory beneath optimization-final-gateup")
    parser.add_argument("--preflight-only", action="store_true", help="Verify staged runtime ABI/exports only; do not benchmark or initialize a CPU backend")
    parser.add_argument("--reuse-preflight", action="store_true", help="Use previously verified staged runtime after checking its hashes and helper headers")
    args = parser.parse_args()
    allowed = {"IQ2_XXS", "IQ2_XS", "IQ2_S", "IQ3_S", "IQ3_XXS", "Q2_0"}
    if not 10 <= args.iterations <= 100 or not 10 <= args.timeout <= 300 or not set(args.formats) <= allowed:
        parser.error("Format or bounded runtime option is invalid")
    exe = args.executable.resolve()
    if not exe.exists():
        parser.error("Compile tools/gateup_format_bench.cpp into the kernels folder first")
    folder = args.output.resolve()
    if not folder.is_relative_to(FOLDER.parent.resolve()):
        parser.error("Output must stay beneath data/optimization-final-gateup")
    if args.runtime_bin and not args.runtime_source:
        parser.error("--runtime-bin requires --runtime-source for the header ABI check")
    if args.preflight_only and not args.runtime_bin:
        parser.error("--preflight-only requires an explicit --runtime-bin")
    folder.mkdir(parents=True, exist_ok=True)
    runtime_bin = BIN.resolve()
    runtime_receipt = None
    if args.reuse_preflight:
        if args.runtime_bin or args.runtime_source or args.preflight_only:
            parser.error("Reuse the existing receipt without staging a different source/runtime")
        runtime_receipt = json.loads((folder / "runtime-preflight.json").read_text(encoding="utf-8"))
        if runtime_receipt["status"] != "ready":
            raise ValueError("Saved runtime preflight did not pass")
        runtime_bin = Path(runtime_receipt["staged_runtime"]).resolve()
        exe = Path(runtime_receipt["staged_helper"]).resolve()
        if not runtime_bin.is_relative_to(folder) or not exe.is_relative_to(folder):
            raise ValueError("Saved runtime paths outside phase output")
        if digest(exe) != runtime_receipt["helper_sha256"]:
            raise ValueError("Saved helper changed since ABI preflight")
        for name, receipt in runtime_receipt["dll_receipts"].items():
            if digest(runtime_bin / name) != receipt["sha256"]:
                raise ValueError("Saved runtime DLL changed since ABI preflight")
        for relative, receipt in runtime_receipt["header_abi"].items():
            if digest(ROOT / "third_party/llama.cpp" / relative) != receipt["helper_header_sha256"]:
                raise ValueError("Helper source ABI changed since saved preflight")
    elif args.runtime_bin:
        exe, runtime_bin, runtime_receipt = stage_runtime(args.runtime_bin.resolve(), args.runtime_source.resolve(), exe, folder)
    if args.preflight_only:
        print(json.dumps({"status": "ready", "receipt": str(folder / "runtime-preflight.json"), "benchmarks_run": 0}), flush=True)
        return
    inventory = subprocess.run(["powershell", "-NoProfile", "-Command",
        "Get-Process -Name llama-server,llama-imatrix,llama-bench,llama-perplexity -ErrorAction SilentlyContinue | Select-Object -ExpandProperty ProcessName"],
        capture_output=True, text=True, check=False)
    if inventory.stdout.strip():
        raise RuntimeError("A model worker is active. Coordinate before running CPU preflight: " + inventory.stdout.strip())
    if (folder / "summary.json").exists():
        raise FileExistsError("Existing results are preserved; choose a fresh --output directory")
    lock = folder / "preflight.lock"
    try:
        lock_fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as error:
        raise RuntimeError("A preflight lock exists; verify its worker before retrying") from error
    os.write(lock_fd, str(os.getpid()).encode())
    os.close(lock_fd)
    environment = os.environ.copy()
    environment["PATH"] = str(runtime_bin) + os.pathsep + environment.get("PATH", "")
    results = []
    common = {"scope": "Synthetic actual CPU routed gate/up kernel; not model tokens/s or model quality",
        "shape": [2560, 640, 512], "synthetic_templates": 10, "populated_experts": 512, "routes_per_token": 10,
        "threads": 12, "warmups_per_group": 10, "groups": 5, "iterations_per_group": args.iterations,
        "importance": "Uniform positive per-input importance, synthetic normal weights; no corpus calibration",
        "helper_sha256": digest(exe), "source_sha256": digest(ROOT / "tools/gateup_format_bench.cpp"),
        "dll_sha256": {name: digest(runtime_bin / name) for name in ("ggml-base.dll", "ggml-cpu.dll")},
        "runtime_bin": str(runtime_bin), "runtime_preflight": runtime_receipt,
        "started_utc": datetime.now(timezone.utc).isoformat(), "pid": os.getpid()}
    write(folder / "status.json", {**common, "status": "running"})
    try:
        for name in args.formats:
            kind = getattr(gguf.GGMLQuantizationType, name)
            block, packed_size = gguf.GGML_QUANT_SIZES[kind]
            if 2560 % block:
                raise ValueError(f"Incompatible gate/up row dimension: {name}")
            for layout in args.layouts:
                write(folder / "status.json", {**common, "status": "running", "format": name, "layout": layout})
                output = subprocess.run([str(exe), str(int(kind)), str(args.iterations), layout, str(folder)],
                    capture_output=True, text=True, env=environment, timeout=args.timeout, check=False)
                prefix = folder / f"{name}-{layout}"
                prefix.with_suffix(".jsonl").write_text(output.stdout, encoding="utf-8")
                prefix.with_suffix(".stderr.log").write_text(output.stderr, encoding="utf-8")
                rows = [json.loads(line) for line in output.stdout.splitlines() if line.startswith("{")]
                timing = [row for row in rows if row["kind"] == "timing"]
                packing = [row for row in rows if row["kind"] == "packing"]
                if output.returncode or len(timing) != 5 or len(packing) != 1 or not packing[0]["passed"]:
                    raise RuntimeError(f"Packing/kernel preflight failed: {name}/{layout}, exit {output.returncode}")
                assert packing[0]["expert_bytes"] == 2560 * 640 // block * packed_size
                payload = folder / (packing[0]["format"] + ".selected-experts.bin")
                assert payload.stat().st_size == packing[0]["selected_payload_bytes"]
                milliseconds = [row["milliseconds_per_routed_matmul"] for row in timing]
                result = {"format": name, "layout": layout, "buffer": timing[0]["buffer"],
                    "mean_milliseconds": statistics.mean(milliseconds), "median_milliseconds": statistics.median(milliseconds),
                    "minimum_milliseconds": min(milliseconds), "maximum_milliseconds": max(milliseconds),
                    "group_means_milliseconds": milliseconds, "packing": packing[0],
                    "payload_sha256": digest(payload), "log_sha256": digest(prefix.with_suffix(".jsonl"))}
                results.append(result)
                write(folder / "summary.json", {**common, "results": results, "status": "running"})
                print(json.dumps({"format": name, "layout": layout, "mean_milliseconds": result["mean_milliseconds"],
                    "kernel_rmse": packing[0]["maximum_kernel_relative_rmse"]}), flush=True)
        complete = {**common, "results": results, "status": "complete", "completed_utc": datetime.now(timezone.utc).isoformat(),
            "limitations": ["512 distinct expert slots contain ten repeating synthetic templates; routes cycle over the full weight working set.",
                "Synthetic packing error does not measure model quality or preserve the original GSQ assignment.",
                "CPU DLL identities are recorded; operator timing is not whole-model tokens/s. Confirm with the fixed production model runtime after any accepted build."]}
        write(folder / "summary.json", complete)
        write(folder / "status.json", {**common, "status": "complete", "completed_utc": complete["completed_utc"]})
    except Exception as error:
        write(folder / "status.json", {**common, "status": "failed", "error": str(error), "finished_utc": datetime.now(timezone.utc).isoformat()})
        raise
    finally:
        lock.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
