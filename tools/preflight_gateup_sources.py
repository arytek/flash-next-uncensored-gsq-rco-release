"""Read-only gate/up source, layout and exact payload-size preflight.

No model inference, downloads, source edits, or whole-model hashing. Only the
small JSON evidence report is written. Source headers must be fetched and checked
before their unavailable BF16 payloads can be used.
"""
from __future__ import annotations

import hashlib
import json
import math
import struct
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "third_party/llama.cpp/gguf-py"))
import gguf  # noqa: E402

SOURCE = ROOT / "data/source"
REVISION = "deb02632504bb214702bc28b0381a93d3112f500"
REPO = "windowsxp811203/Qwen3.8-Flash-Next-Abliterated"
CONTROL = Path(r"models\flash-next-uncensored-gsq-rco-models\Qwen3.8-Flash-Next-Abliterated-GSQ-RCO-IQ3_XXS-Calibrated-00001-of-00002.gguf")


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def small_digest(path: Path):
    assert path.stat().st_size < 16 * 1024 * 1024
    return hashlib.sha256(path.read_bytes()).hexdigest()


def header(path: Path):
    with path.open("rb") as stream:
        size = struct.unpack("<Q", stream.read(8))[0]
        assert 0 < size < 16 * 1024 * 1024
        raw = stream.read(size)
        assert len(raw) == size
    return read_header(raw), 8 + size, hashlib.sha256(raw).hexdigest()


def read_header(raw: bytes):
    return json.loads(raw)


def main():
    manifest = read(SOURCE / "text-source-manifest.json")
    assert manifest["revision"] == REVISION and manifest["complete"]
    index = read(SOURCE / "model.safetensors.index.json")["weight_map"]
    tree = read(SOURCE / ".cache/huggingface/trees" / f"{REVISION}.json")["files"]
    config = read(SOURCE / "config.json")["text_config"]
    assert config["num_hidden_layers"] == 48
    experts, intermediate, hidden = (config[k] for k in ("num_experts", "moe_intermediate_size", "hidden_size"))
    assert (experts, intermediate, hidden) == (512, 640, 2560)
    expected_hf_shape = [experts, 2 * intermediate, hidden]
    expected_ggml_shape = [hidden, intermediate, experts]
    per_half_elements = experts * intermediate * hidden
    edited = {entry["tensor"] for entry in read(SOURCE / "ABLIT_META.json")["stats"]}
    reader = gguf.GGUFReader(CONTROL)
    assert reader.get_field("general.architecture").contents() == "qwen4exp"
    tensors = {tensor.name: tensor for tensor in reader.tensors}
    receipt = read(ROOT / "data/comparison-sc117-20261006/file-receipts.json")[str(CONTROL)]
    assert CONTROL.stat().st_size == receipt["bytes"]
    assert CONTROL.stat().st_mtime_ns == receipt["mtime_ns"]
    tensor_map = gguf.get_tensor_name_map(gguf.MODEL_ARCH.QWEN4EXP, 48)
    source_paths = [SOURCE]
    source_paths += list((ROOT / "data/.hf-cache/hub/models--windowsxp811203--Qwen3.8-Flash-Next-Abliterated/snapshots").glob(REVISION))
    source_paths += list((Path.home() / ".cache/huggingface/hub/models--windowsxp811203--Qwen3.8-Flash-Next-Abliterated/snapshots").glob(REVISION))
    rows = []
    counts = Counter()
    payload_total = 0
    for layer in range(48):
        name = f"model.language_model.layers.{layer}.mlp.experts.gate_up_proj"
        assert name in index and name not in edited
        filename = index[name]
        available = next((root / filename for root in source_paths if (root / filename).is_file()), None)
        record = {"layer": layer, "hf_tensor": name, "hf_file": filename,
                  "pinned_file": tree[filename], "expected_hf_shape": expected_hf_shape,
                  "expected_bf16_payload_bytes": 2 * math.prod(expected_hf_shape),
                  "edited_by_ablation": False, "local_bf16_file": str(available) if available else None,
                  "source_header_verified": False, "source_tensor_byte_range": None,
                  "gguf_tensors": []}
        if available:
            contents, data_start, digest = header(available)
            item = contents[name]
            assert item["dtype"] == "BF16" and item["shape"] == expected_hf_shape
            start, end = item["data_offsets"]
            assert end - start == record["expected_bf16_payload_bytes"]
            assert data_start + end <= available.stat().st_size
            record.update(source_header_verified=True, source_header_sha256=digest,
                          source_tensor_byte_range=[data_start + start, data_start + end - 1])
        for half in ("gate", "up"):
            target = f"blk.{layer}.ffn_{half}_exps.weight"
            mapped = tensor_map.get_name(f"model.layers.{layer}.mlp.experts.{half}_proj.weight", try_suffixes=(".weight",))
            assert mapped == target
            tensor = tensors[target]
            assert tensor.shape.tolist() == expected_ggml_shape
            counts[tensor.tensor_type.name] += 1
            payload_total += int(tensor.n_bytes)
            block, size = gguf.GGML_QUANT_SIZES[tensor.tensor_type]
            assert hidden % block == 0
            assert int(tensor.n_bytes) == per_half_elements // block * size
            record["gguf_tensors"].append({"name": target, "shape_ggml": tensor.shape.tolist(),
                "format": tensor.tensor_type.name, "payload_bytes": int(tensor.n_bytes),
                "data_offset": int(tensor.data_offset)})
        rows.append(record)
    # Sentinel confirms the split recipe is contiguous and reversible. This does
    # not substitute for a real BF16 source/header and prediction comparison.
    sentinel = np.arange(2 * 8 * 10).reshape(2, 8, 10)
    gate, up = np.ascontiguousarray(sentinel[:, :4]), np.ascontiguousarray(sentinel[:, 4:])
    assert np.array_equal(np.concatenate((gate, up), axis=1), sentinel)
    lookup_bytes = 28800138432
    control_total = receipt["bytes"] + lookup_bytes
    sizes = {}
    for fmt in ("IQ2_XXS", "IQ2_XS", "IQ2_S", "IQ3_XXS", "IQ3_S", "IQ4_NL", "Q2_0"):
        kind = getattr(gguf.GGMLQuantizationType, fmt)
        block, block_bytes = gguf.GGML_QUANT_SIZES[kind]
        compatible = hidden % block == 0
        new_payload = per_half_elements // block * block_bytes * 96 if compatible else None
        sizes[fmt] = {"block_values": block, "block_bytes": block_bytes,
                      "row_width_compatible": compatible,
                      "all_gate_up_payload_bytes": new_payload,
                      "projected_total_bytes": control_total - payload_total + new_payload if compatible else None,
                      "saving_bytes": payload_total - new_payload if compatible else None}
    source_files = {row["hf_file"]: row["pinned_file"] for row in rows}
    missing = [row for row in rows if not row["local_bf16_file"]]
    converter_paths = ["third_party/llama.cpp/conversion/qwen4exp.py", "third_party/llama.cpp/conversion/qwen.py",
                       "third_party/llama.cpp/conversion/base.py", "third_party/llama.cpp/gguf-py/gguf/tensor_mapping.py"]
    report = {"utc": datetime.now(timezone.utc).isoformat(), "status": "source-preflight-complete",
        "privacy": "Local audit only; no downloads, inference, external messages or model edits",
        "source_repository": REPO, "source_revision": REVISION,
        "source_index_sha256": small_digest(SOURCE / "model.safetensors.index.json"),
        "ablation_meta_sha256": small_digest(SOURCE / "ABLIT_META.json"),
        "cache_directories_checked": [str(path) for path in source_paths],
        "bf16_gate_up_tensors_local": 48 - len(missing), "bf16_gate_up_tensors_missing": len(missing),
        "full_missing_source_file_bytes": sum(row["pinned_file"]["size"] for row in missing),
        "full_gate_up_bf16_payload_bytes": 48 * 2 * math.prod(expected_hf_shape),
        "warning": "Fresh BF16 gate/up payloads are not local. Do not dequantize retained ISTA as a substitute for fresh high precision source." if missing else None,
        "mapping": {"inheritance": "Qwen4ExpTextModel -> _LinearAttentionVReorderBase -> Qwen3NextModel -> Qwen2MoeModel",
            "prefix": "filter_tensors removes language_model. from the HF name",
            "split": "BF16 [512,1280,2560] splits axis -2 at640; first640 rows gate, final640 up",
            "output_contiguous_shape": [512,640,2560], "ggml_shape": expected_ggml_shape,
            "head_permutation": "None for these expert gate/up weights; V-head reordering targets linear attention projections only",
            "tensor_name_mapping_checked": True, "sentinel_split_roundtrip_passed": True,
            "real_bf16_header_layout_checks": "Pending for all unavailable source files",
            "converter_fingerprints": {path: small_digest(ROOT / path) for path in converter_paths}},
        "control_identity": {**receipt, "verification": "Previously full SHA verified; size/mtime unchanged, no new huge hash"},
        "control_gate_up_tensor_count": 96, "control_gate_up_formats": dict(counts),
        "control_gate_up_payload_bytes": payload_total, "control_total_bytes": control_total,
        "format_size_arithmetic": sizes,
        "size_scope": "Exact packed tensor payload arithmetic. Same header length and tensor alignment assumed; no quality or speed prediction.",
        "pilot_fetch": {"suggested_layers": [8,28,47], "experts_per_layer": 3,
            "per_expert_fused_bf16_bytes": 2 * 2 * intermediate * hidden,
            "nine_expert_fused_bf16_bytes": 9 * 2 * 2 * intermediate * hidden,
            "full_three_source_shards_bytes": sum(row["pinned_file"]["size"] for row in rows if row["layer"] in (8,28,47)),
            "full_shard_receipts": [{"layer": row["layer"], "file": row["hf_file"],
                "bytes": row["pinned_file"]["size"], "sha256": row["pinned_file"]["lfs_sha256"],
                "hf_tensor": row["hf_tensor"],
                "url": f"https://huggingface.co/{REPO}/resolve/{REVISION}/{row['hf_file']}"}
                for row in rows if row["layer"] in (8,28,47)],
            "steps": ["Read8-byte safetensors header length from each pinned shard; require exact206 Content-Range",
                      "Read and validate JSON header/dtype/shape/offsets against pinned shard size",
                      "Fetch bounded expert/row payload ranges, verify exact bounds and finite BF16 data; save SHA/receipt per range",
                      "Verify split against actual source and original engine before building a whole candidate"],
            "availability": "Byte-range headers for missing files have not been read; exact absolute tensor offsets are intentionally unresolved"},
        "source_files": source_files, "layers": rows}
    folder = ROOT / "data/optimization-final-gateup"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "source-preflight.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ["status", "bf16_gate_up_tensors_local", "bf16_gate_up_tensors_missing",
          "full_missing_source_file_bytes", "control_gate_up_formats", "control_gate_up_payload_bytes", "format_size_arithmetic"]}))


if __name__ == "__main__":
    main()
