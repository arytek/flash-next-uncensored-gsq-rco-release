"""Fetch only the abliterated GGUF tensors needed by the hybrid build.

The resulting GGUF is sparse and INCOMPLETE by design. Never publish it or use it
for inference: all unselected tensor payloads are holes. Use it only as the
`--donor` input to assemble_hybrid.py after this command reports success.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import gguf

from tools.assemble_hybrid import edited_names

REVISION = "c3365c410baa29bdd3d7cc8cbc2bf9bee0de2f3a"
SOURCE_URL = ("https://huggingface.co/windowsxp811203/"
              "Qwen3.8-Flash-Next-Abliterated-GGUF/resolve/" + REVISION + "/"
              "Qwen3.8-Flash-Next-Abliterated-Q4_K_M.gguf")
SOURCE_SIZE = 119150722112


def make_sparse(path: Path, size: int, header: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.stat().st_size != size:
            raise ValueError(f"Existing sparse file has wrong size: {path}")
        with path.open("rb") as stream:
            if stream.read(len(header)) != header:
                raise ValueError("Existing sparse file has a different GGUF header")
        return
    path.touch()
    if sys.platform == "win32":
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.CreateFileW.restype = ctypes.c_void_p
        kernel.CreateFileW.argtypes = [ctypes.c_wchar_p, ctypes.c_uint32, ctypes.c_uint32,
                                        ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint32,
                                        ctypes.c_void_p]
        kernel.DeviceIoControl.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_void_p,
                                            ctypes.c_uint32, ctypes.c_void_p, ctypes.c_uint32,
                                            ctypes.POINTER(ctypes.c_uint32), ctypes.c_void_p]
        kernel.SetFilePointerEx.argtypes = [ctypes.c_void_p, ctypes.c_longlong,
                                             ctypes.c_void_p, ctypes.c_uint32]
        kernel.SetEndOfFile.argtypes = [ctypes.c_void_p]
        kernel.CloseHandle.argtypes = [ctypes.c_void_p]
        handle = kernel.CreateFileW(str(path.resolve()), 0x40000000, 3, None, 3, 0, None)
        if handle == ctypes.c_void_p(-1).value:
            raise ctypes.WinError(ctypes.get_last_error())
        try:
            returned = ctypes.c_uint32()
            if not kernel.DeviceIoControl(handle, 0x900C4, None, 0, None, 0,
                                          ctypes.byref(returned), None):
                raise ctypes.WinError(ctypes.get_last_error())
            if not kernel.SetFilePointerEx(handle, size, None, 0) or not kernel.SetEndOfFile(handle):
                raise ctypes.WinError(ctypes.get_last_error())
        finally:
            kernel.CloseHandle(handle)
    else:
        with path.open("r+b") as stream:
            stream.truncate(size)
    with path.open("r+b") as stream:
        stream.write(header)


def fetch_one(path: Path, name: str, start: int, size: int) -> dict:
    end = start + size - 1
    request = urllib.request.Request(SOURCE_URL, headers={"Range": f"bytes={start}-{end}"})
    for attempt in range(4):
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                content_range = response.headers.get("Content-Range", "")
                expected = f"bytes {start}-{end}/{SOURCE_SIZE}"
                if response.status != 206 or content_range != expected:
                    raise ValueError(f"Unexpected range response: {response.status}, {content_range}")
                digest = hashlib.sha256()
                received = 0
                with path.open("r+b", buffering=0) as target:
                    target.seek(start)
                    while received < size:
                        block = response.read(min(8 * 1024 * 1024, size - received))
                        if not block:
                            raise ValueError(f"Truncated response: {received}/{size}")
                        target.write(block)
                        digest.update(block)
                        received += len(block)
                    os.fsync(target.fileno())
                return {"name": name, "offset": start, "bytes": size, "sha256": digest.hexdigest()}
        except Exception as error:
            if attempt == 3:
                # Avoid leaking the signed CDN redirect URL in error messages.
                raise RuntimeError(f"Range fetch failed for {name}: {type(error).__name__}") from None
            time.sleep(2 ** attempt)
    raise AssertionError("unreachable")


def verify_ranges(path: Path, tensors: dict[str, tuple[int, int]], completed: dict) -> None:
    if set(completed) != set(tensors):
        raise ValueError("Range manifest does not cover all edited tensors")
    with path.open("rb") as stream:
        for name, (start, size) in tensors.items():
            entry = completed[name]
            if entry["offset"] != start or entry["bytes"] != size:
                raise ValueError(f"Range manifest offset or length differs: {name}")
            stream.seek(start)
            digest = hashlib.sha256()
            remaining = size
            while remaining:
                block = stream.read(min(8 * 1024 * 1024, remaining))
                if not block:
                    raise ValueError(f"Sparse donor ended inside {name}")
                digest.update(block)
                remaining -= len(block)
            if digest.hexdigest() != entry["sha256"]:
                raise ValueError(f"Sparse donor checksum mismatch: {name}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--header", type=Path, required=True, help="First 32 MiB from a pinned-revision HTTP Range request")
    parser.add_argument("--output", type=Path, required=True, help="Sparse donor GGUF")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--verify-only", action="store_true", help="Check all recorded ranges without downloading")
    args = parser.parse_args()
    if args.workers < 1 or args.workers > 8:
        raise ValueError("Use 1 to 8 concurrent range fetches")
    header = args.header.read_bytes()
    if len(header) < 32 * 1024 * 1024 or header[:4] != b"GGUF":
        raise ValueError("Expected a 32 MiB GGUF header prefix")
    make_sparse(args.output, SOURCE_SIZE, header)
    reader = gguf.GGUFReader(args.output)
    selected = edited_names()
    tensors = {t.name: (int(t.data_offset), int(t.n_bytes)) for t in reader.tensors if t.name in selected}
    if set(tensors) != selected:
        raise ValueError(f"Missing donor tensors: {sorted(selected - set(tensors))}")
    del reader
    manifest_path = args.output.with_suffix(".ranges.json")
    header_sha = hashlib.sha256(header).hexdigest()
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["source_revision"] != REVISION or manifest["header_sha256"] != header_sha:
            raise ValueError("Range manifest does not match this pinned donor")
    else:
        manifest = {"source_revision": REVISION, "source_size": SOURCE_SIZE,
                    "header_sha256": header_sha, "completed": {}}
    todo = sorted(((name, *values) for name, values in tensors.items()
                   if name not in manifest["completed"]), key=lambda item: item[1])
    if args.verify_only:
        verify_ranges(args.output, tensors, manifest["completed"])
        print(f"Verified all {len(tensors)} sparse donor ranges")
        return
    print(f"Fetching {len(todo)} of 146 edited tensors, "
          f"{sum(size for _, _, size in todo) / 2**30:.2f} GiB")
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(fetch_one, args.output, name, start, size): name
                   for name, start, size in todo}
        for future in as_completed(futures):
            result = future.result()
            manifest["completed"][result["name"]] = result
            temporary = manifest_path.with_suffix(".json.tmp")
            temporary.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            temporary.replace(manifest_path)
            print(f"{len(manifest['completed'])}/146 {result['name']}", flush=True)
    if len(manifest["completed"]) != len(selected):
        raise ValueError("Not all edited tensors were fetched")
    verify_ranges(args.output, tensors, manifest["completed"])
    print(f"Sparse donor ready: {args.output}")


if __name__ == "__main__":
    main()
