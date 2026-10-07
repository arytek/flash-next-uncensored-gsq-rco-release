"""Extract complete verified published GSQ tensors for an explicitly selected trial.

Requires a saved human-agent allocation decision. Never chooses precision from
confirmation answers, modifies the control, or accepts a candidate.
"""
from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.assemble_gateup_trial import exclusive_model_lock, PHASE, gguf
from tools.read_gateup_capture import sha256


def main():
    decision = json.loads((PHASE / 'published-allocation-decision.json').read_text(encoding='utf-8-sig'))
    if not decision['pilot_build_selected'] or decision['confirmation_used']:
        raise ValueError('Explicit pilot build decision required; confirmation must remain unused')
    budget = json.loads((PHASE / 'budget.json').read_text(encoding='utf-8-sig'))
    if budget['status'].startswith('complete'):
        raise RuntimeError('Final optimisation phase is closed; no new trial extraction')
    search_deadline = datetime.fromisoformat(budget['deadline_utc']) - timedelta(hours=budget['reserve_hours'])
    def check_deadline():
        if datetime.now(timezone.utc) >= search_deadline:
            raise TimeoutError('Validation reserve reached')
    check_deadline()
    evidence = json.loads((PHASE / 'published-q2-gateup/summary.json').read_text(encoding='utf-8'))
    if evidence['status'] != 'complete' or evidence['confirmation_used']:
        raise ValueError('Completed unconfirmed published-weight pilot required')
    source_identity = evidence['source']
    source = Path(source_identity['path'])
    if source.stat().st_size != source_identity['bytes'] or source.stat().st_mtime_ns != source_identity['mtime_ns']:
        raise ValueError('Fully verified source has changed')
    control_identity = evidence['control']
    control = Path(control_identity['path'])
    if control.stat().st_size != control_identity['bytes'] or control.stat().st_mtime_ns != control_identity['mtime_ns']:
        raise ValueError('Preserved control changed')
    if decision['source_sha256'] != source_identity['sha256'] or decision['control_sha256'] != control_identity['sha256']:
        raise ValueError('Allocation decision does not match verified inputs')
    reader = gguf.GGUFReader(source)
    control_reader = gguf.GGUFReader(control)
    originals = {t.name: t for t in control_reader.tensors}
    selected = {f'blk.{layer}.ffn_{half}_exps.weight' for layer in decision['layers'] for half in ('gate', 'up')}
    tensors = {t.name: t for t in reader.tensors}
    if selected - tensors.keys() or selected - originals.keys():
        raise ValueError('Requested complete tensors absent')
    pool = PHASE / 'published-q2-payloads'
    pool.mkdir(parents=True, exist_ok=True)
    replacements = {}
    for name in sorted(selected, key=lambda key: tensors[key].data_offset):
        check_deadline()
        tensor = tensors[name]
        old = originals[name]
        if tuple(tensor.shape) != (2560, 640, 512) or tuple(old.shape) != tuple(tensor.shape):
            raise ValueError('Published tensor layout differs')
        if tensor.tensor_type.name != 'Q2_0' or old.tensor_type.name != 'IQ3_S':
            raise ValueError('This allocation changes only IQ3_S gate/up to published Q2_0')
        path = pool / (name.replace('.', '_') + '.Q2_0.bin')
        if path.exists():
            raise FileExistsError('Existing extraction needs review before rerun: ' + str(path))
        digest = hashlib.sha256()
        with source.open('rb') as inp, path.open('xb') as output:
            inp.seek(tensor.data_offset)
            remaining = tensor.n_bytes
            while remaining:
                check_deadline()
                block = inp.read(min(8 * 1024**2, remaining))
                if not block:
                    raise IOError('Incomplete published tensor read')
                output.write(block)
                digest.update(block)
                remaining -= len(block)
        if path.stat().st_size != tensor.n_bytes or sha256(path) != digest.hexdigest():
            raise ValueError('Extracted tensor checksum failed')
        replacements[name] = {'path': str(path), 'format': 'Q2_0', 'experts': 512, 'shape_ggml': [2560, 640, 512],
            'bytes': tensor.n_bytes, 'sha256': digest.hexdigest(), 'quantization_method': 'Retained published ISTA GSQ-RCO; no local requantization',
            'source': {**source_identity, 'tensor': name, 'tensor_data_offset': tensor.data_offset}, 'prior_control_format': old.tensor_type.name}
    result = {'feasible': True, 'pilot_build_selected': True, 'development_selected': False, 'confirmation_used': False,
        'selection_basis': decision['selection_basis'], 'decision_sha256': sha256(PHASE / 'published-allocation-decision.json'),
        'control': control_identity, 'replacements': replacements, 'source': source_identity, 'models_accepted': 0,
        'scope': 'Whole-model private trial awaiting speed, development, confirmation, coding, behavior and memory gates'}
    (PHASE / 'published-q2-allocation.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'status': 'extracted-unaccepted-trial', 'tensors': len(replacements), 'new_m_bytes': sum(t['bytes'] for t in replacements.values())}))


if __name__ == '__main__':
    with exclusive_model_lock():
        main()
