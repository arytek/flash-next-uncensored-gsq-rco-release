"""Isolate refined IQ4 changes while retaining the control's two Q2 tensors."""
import hashlib
import json
from pathlib import Path

from tools.assemble_refined_trial import predicted_first_shard_bytes, gguf

ROOT = Path(__file__).resolve().parents[1]


def main():
    folder = ROOT / 'data/optimization-48h'
    source_path = folder / 'allocation-54344308416.json'
    source = json.loads(source_path.read_text())
    retained = ['blk.5.ffn_down_exps.weight', 'blk.13.ffn_down_exps.weight']
    if not source['feasible'] or any(source['replacement_formats'][n] != 'Q2_0' for n in retained):
        raise ValueError('Expected balanced allocation with Q2 layers 5 and 13')
    replacements = {n: fmt for n, fmt in source['replacement_formats'].items() if n not in retained}
    if len(replacements) != 46 or set(replacements.values()) != {'IQ4_NL'}:
        raise ValueError('Unexpected IQ4 layout')
    control = gguf.GGUFReader(source['control'])
    tensors = {t.name: t for t in control.tensors}
    if any(tensors[n].tensor_type != gguf.GGMLQuantizationType.Q2_0 for n in retained):
        raise ValueError('Control Q2 tensor type differs')
    output = folder / 'allocation-q2-backoff.json'
    predicted, header = predicted_first_shard_bytes(control, replacements, output.with_suffix('.header.bin'))
    if predicted != source['predicted_transformer_bytes']:
        raise ValueError('Backoff changed expected byte size')
    plan = {
        'feasible': True,
        'control': source['control'],
        'variants': str((ROOT / source['variants']).resolve()),
        'replacement_formats': replacements,
        'retained_control_tensors': retained,
        'retained_control_transformer_sha256': '84512ed5aaa14930c56345eaaa88adccddce5a80a700b102c0b9fbfdc955f507',
        'transformer_budget_bytes': source['transformer_budget_bytes'],
        'predicted_transformer_bytes': predicted,
        'header_bytes': header,
        'predicted_total_bytes': source['predicted_total_bytes'],
        'formats': {'IQ4_NL': 46, 'Q2_0': 2},
        'mandatory_iq4_layers': source['mandatory_iq4_layers'],
        'source_allocation_sha256': hashlib.sha256(source_path.read_bytes()).hexdigest(),
        'objective': 'Diagnostic candidate: refined balanced IQ4 with original control Q2 layers 5 and 13. Same formats and bytes. Isolate Q2 refinement effects; answer tests decide quality.',
        'metric_note': 'Previous selected-payload layer scores omitted because two payloads revert to the control. No whole-model quality gain is predicted.',
    }
    output.write_text(json.dumps(plan, indent=2) + '\n')
    print(json.dumps({'allocation': str(output), 'transformer_bytes': predicted, 'total_bytes': plan['predicted_total_bytes'], 'retained_control_tensors': retained}))


if __name__ == '__main__':
    main()
