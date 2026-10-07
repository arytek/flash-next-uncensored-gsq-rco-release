"""Assemble one private GGUF from complete, verified gate/up tensor payloads.

Every expert-down, dense and lookup payload stays byte-identical to control.
Dry runs write only a small metadata preview on M:. No uploads or overwrites.
"""
from __future__ import annotations
import argparse
import ctypes
from contextlib import contextmanager
import hashlib
import json
import math
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'third_party/llama.cpp/gguf-py'))
import gguf
from tools.assemble_hybrid import copy_metadata
from tools.verify_calibrated import equal_ranges
from tools.read_gateup_capture import sha256

PHASE = ROOT / 'data/optimization-final-gateup'
TRIAL_ROOT = Path('models/flash-next-uncensored-gsq-rco-models/trials/final-gateup-20261007')
FORMATS = {'IQ2_S', 'IQ2_XS', 'IQ2_XXS', 'IQ3_XXS', 'Q2_0'}
DESCRIPTION = ('Private compact trial: retained corrected abliterated control weights; selected complete gate/up tensors freshly packed '
               'from unchanged BF16 source with native calibrated IQ codes and optional training-only global-scale refinement, or retained '
               'from published ISTA GSQ-RCO variants as identified in the allocation provenance. Evaluated against a routed mixed block '
               'reference. All expert-down, dense and lookup payloads retained. Targeted GSQ-based allocation, '
               'not full upstream GSQ/RCO, complete BF16 reference or accepted public release.')


@contextmanager
def exclusive_model_lock():
    """Share the Windows exclusive lock used by capture/validation workers."""
    from ctypes import wintypes
    api = ctypes.WinDLL('kernel32', use_last_error=True)
    api.CreateFileW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD,
                               wintypes.LPVOID, wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE]
    api.CreateFileW.restype = wintypes.HANDLE
    api.CloseHandle.argtypes = [wintypes.HANDLE]
    api.CloseHandle.restype = wintypes.BOOL
    path = ROOT / 'data/optimization-48h/candidate-validation.lock'
    handle = api.CreateFileW(str(path), 0xC0000000, 0, None, 4, 0x80, None)
    if handle == ctypes.c_void_p(-1).value:
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        yield
    finally:
        api.CloseHandle(handle)


def writer_for(source, path):
    writer = gguf.GGUFWriter(path, 'qwen4exp')
    copy_metadata(writer, source, 'Compact gate-up', DESCRIPTION)
    for name, field in source.fields.items():
        if name.startswith('split.'):
            writer.add_key_value(name, field.contents(), field.types[0])
    writer.data_alignment = int(source.alignment)
    return writer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--allocation', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    if not args.allocation.resolve().is_relative_to(PHASE.resolve()):
        raise ValueError('Allocation must be a private phase record')
    budget = json.loads((PHASE / 'budget.json').read_text(encoding='utf-8-sig'))
    if budget['status'].startswith('complete'):
        raise RuntimeError('Final optimisation phase is closed; no more trial builds')
    deadline = datetime.fromisoformat(budget['deadline_utc'])
    from datetime import timedelta
    search_deadline = deadline - timedelta(hours=budget['reserve_hours'])
    def check_deadline():
        if datetime.now(timezone.utc) >= search_deadline:
            raise TimeoutError('Build reserve reached; no additional candidate assembly')
    check_deadline()
    plan = json.loads(args.allocation.read_text(encoding='utf-8-sig'))
    if not plan.get('feasible') or not (plan.get('development_selected') or plan.get('pilot_build_selected')):
        raise ValueError('Allocation must document a feasible pilot or development build decision')
    if plan.get('confirmation_used'):
        raise ValueError('Do not tune or rebuild from confirmation answers')
    control_identity = plan['control']
    control = Path(control_identity['path'])
    preserved = json.loads((PHASE / 'source-preflight.json').read_text(encoding='utf-8-sig'))['control_identity']
    if control_identity != preserved or control.stat().st_size != preserved['bytes'] or control.stat().st_mtime_ns != preserved['mtime_ns']:
        raise ValueError('Allocation does not use the preserved control')
    if not args.output.name.endswith('-00001-of-00002.gguf') or args.output.exists():
        raise ValueError('Choose an unused first-of-two shard name')
    if not args.output.resolve().is_relative_to(TRIAL_ROOT.resolve()) or args.output.resolve() == control.resolve():
        raise ValueError('Output must stay inside the dedicated C trial directory')
    source = gguf.GGUFReader(control)
    original = {tensor.name: tensor for tensor in source.tensors}
    replacements = plan['replacements']
    if not replacements or replacements.keys() - original.keys():
        raise ValueError('Replacement names absent or unknown')
    for name, entry in replacements.items():
        check_deadline()
        tensor = original[name]
        if not name.endswith(('.ffn_gate_exps.weight', '.ffn_up_exps.weight')) or tuple(tensor.shape) != (2560, 640, 512):
            raise ValueError('Only complete gate/up expert tensors may change')
        fmt = entry['format']
        if fmt not in FORMATS or entry['experts'] != 512 or entry['shape_ggml'] != [2560, 640, 512]:
            raise ValueError('Unsupported or incomplete replacement')
        path = Path(entry['path'])
        if not path.resolve().is_relative_to(PHASE.resolve()):
            raise ValueError('Replacement payload outside the private phase')
        block, size = gguf.GGML_QUANT_SIZES[gguf.GGMLQuantizationType[fmt]]
        expected = 512 * 640 * 2560 // block * size
        if entry['bytes'] != expected or path.stat().st_size != expected or sha256(path) != entry['sha256']:
            raise ValueError('Replacement byte count or checksum failed')
    preview = PHASE / 'assembly-preview.gguf'
    writer = writer_for(source, preview)
    packed = 0
    for tensor in source.tensors:
        shape = tuple(int(value) for value in reversed(tensor.shape))
        entry = replacements.get(tensor.name)
        qtype = gguf.GGMLQuantizationType[entry['format']] if entry else tensor.tensor_type
        size = entry['bytes'] if entry else tensor.n_bytes
        writer.add_tensor_info(tensor.name, shape, np.dtype(np.float32), size, raw_dtype=qtype)
        packed += gguf.GGUFWriter.ggml_pad(size, writer.data_alignment)
    writer.write_header_to_file(); writer.write_kv_data_to_file(); writer.write_ti_data_to_file()
    predicted = gguf.GGUFWriter.ggml_pad(writer.fout[0].tell(), writer.data_alignment) + packed
    writer.close()
    lookup = Path(str(control).replace('-00001-of-00002.gguf', '-00002-of-00002.gguf'))
    if predicted > budget['c_trial_transformer_ceiling_bytes']:
        raise ValueError('Transformer trial exceeds the fixed byte ceiling')
    if (not args.dry_run or 'predicted_transformer_bytes' in plan) and plan.get('predicted_transformer_bytes') != predicted:
        raise ValueError('Exact allocation byte prediction differs from written metadata/layout')
    result = {'scope': 'Private unaccepted trial; no uploads', 'allocation_sha256': sha256(args.allocation),
        'control': control_identity, 'predicted_transformer_bytes': predicted, 'lookup_bytes': lookup.stat().st_size,
        'total_bytes': predicted + lookup.stat().st_size, 'replacement_formats': {name: entry['format'] for name, entry in replacements.items()},
        'dry_run': args.dry_run, 'builder_sha256': sha256(Path(__file__)), 'metadata_description': DESCRIPTION}
    if args.dry_run:
        (PHASE / 'assembly-dry-run.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
        print(json.dumps(result, indent=2))
        return
    if TRIAL_ROOT.exists() and any(TRIAL_ROOT.rglob('*.gguf')):
        raise FileExistsError('An existing C trial must be reviewed before assembling another')
    history_path = PHASE / 'assembly-history.jsonl'
    history = [json.loads(line) for line in history_path.read_text(encoding='utf-8').splitlines()] if history_path.exists() else []
    if len(history) >= budget['max_trial_builds']:
        raise ValueError('The approved phase trial-build ceiling has been reached')
    if shutil.disk_usage('C:/').free - predicted < budget['minimum_c_free_after_build_bytes']:
        raise OSError('Insufficient C free space after the projected trial')
    check_deadline()
    with history_path.open('a', encoding='utf-8') as stream:
        stream.write(json.dumps({'started_utc': datetime.now(timezone.utc).isoformat(), 'output': str(args.output),
                                'allocation_sha256': result['allocation_sha256'], 'predicted_bytes': predicted}) + '\n')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = writer_for(source, args.output)
    for tensor in source.tensors:
        entry = replacements.get(tensor.name)
        if entry:
            fmt = gguf.GGMLQuantizationType[entry['format']]
            block, size = gguf.GGML_QUANT_SIZES[fmt]
            raw = np.memmap(entry['path'], dtype=np.uint8, mode='r', shape=(512, 640, 2560 // block * size))
            writer.add_tensor(tensor.name, raw, raw_dtype=fmt)
        else:
            writer.add_tensor(tensor.name, tensor.data, raw_dtype=tensor.tensor_type)
    writer.write_header_to_file(); writer.write_kv_data_to_file(); writer.write_tensors_to_file(progress=True); writer.close()
    if args.output.stat().st_size != predicted:
        raise ValueError('Actual first-shard bytes differ from exact preview')
    check_deadline()
    target_lookup = Path(str(args.output).replace('-00001-of-00002.gguf', '-00002-of-00002.gguf'))
    os.link(lookup, target_lookup)
    output = gguf.GGUFReader(args.output)
    if {tensor.name for tensor in output.tensors} != original.keys():
        raise ValueError('Written model changed tensor names')
    for tensor in output.tensors:
        old = original[tensor.name]
        entry = replacements.get(tensor.name)
        expected_type = gguf.GGMLQuantizationType[entry['format']] if entry else old.tensor_type
        path = Path(entry['path']) if entry else control
        offset = 0 if entry else old.data_offset
        if tuple(tensor.shape) != tuple(old.shape) or tensor.tensor_type != expected_type:
            raise ValueError('Written tensor shape or type changed unexpectedly')
        if not equal_ranges(path, offset, args.output, tensor.data_offset, tensor.n_bytes):
            raise ValueError('Written tensor payload differs from source: ' + tensor.name)
    result.update({'dry_run': False, 'output_shards': [str(args.output), str(target_lookup)],
        'sha256': [sha256(args.output), sha256(target_lookup)], 'verified_tensors': len(output.tensors),
        'lookup': 'Hardlink to unchanged control lookup', 'completed_utc': datetime.now(timezone.utc).isoformat()})
    args.output.with_suffix('.trial.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'status': 'verified-private-trial', 'total_bytes': result['total_bytes']}))


if __name__ == '__main__':
    with exclusive_model_lock():
        main()
