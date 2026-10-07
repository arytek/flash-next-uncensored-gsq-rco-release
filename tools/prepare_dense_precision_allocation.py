"""Exact-byte plan for control expert weights with F16 edited projections."""
import hashlib
import json
from pathlib import Path
import sqlite3

from tools.assemble_refined_trial import CONTROL, gguf, predicted_first_shard_bytes

ROOT = Path(__file__).resolve().parents[1]


def main():
    folder = ROOT / 'data/optimization-48h'
    variants = folder / 'dense-f16-variants'
    status = json.loads((variants / 'status.json').read_text())
    if status['status'] != 'complete' or len(status['tensors']) != 48 or status['completed_chunks'] != 960:
        raise ValueError('All 48 output projection variants must pass the source/layout gates')
    source = gguf.GGUFReader(CONTROL)
    replacements = {name: 'F16' for name in status['tensors']}
    tensors = {t.name: t for t in source.tensors}
    if any(tensors[name].tensor_type != gguf.GGMLQuantizationType.Q8_0 for name in replacements):
        raise ValueError('Expected control Q8 output projections')
    connection = sqlite3.connect(variants / 'state.sqlite')
    for name, metrics in connection.execute('SELECT name,metrics FROM chunks'):
        row = json.loads(metrics)
        if name not in replacements or not row['q8_control_reconstructed_exactly'] or row['selected_sse'] > row['q8_sse']:
            raise ValueError('F16 source/layout or reconstruction gate failed')
    connection.close()
    output = folder / 'allocation-dense-f16.json'
    description = 'Local mixed trial: retained ISTA GSQ-RCO tensors and corrected abliterated control expert weights. Only 48 edited attention/SSM output projections promoted from Q8_0 to F16 from pinned BF16 source; exact control Q8 reconstruction verified before promotion. Targeted adaptation, not full upstream GSQ/RCO.'
    predicted, header = predicted_first_shard_bytes(source, replacements, output.with_suffix('.header.bin'), description)
    delta = sum(tensors[name].n_elements * 2 - tensors[name].n_bytes for name in replacements)
    if delta != 707788800 or predicted != CONTROL.stat().st_size + delta + header - min(t.data_offset for t in source.tensors):
        raise ValueError('Projection precision bytes disagree with serialized header prediction')
    lookup = Path(str(CONTROL).replace('-00001-of-00002', '-00002-of-00002'))
    plan = {
        'feasible': True, 'control': str(CONTROL), 'variants': str(variants),
        'replacement_formats': replacements,
        'retained_control_transformer_sha256': status['metadata']['control_sha256'],
        'transformer_budget_bytes': predicted, 'predicted_transformer_bytes': predicted,
        'predicted_total_bytes': predicted + lookup.stat().st_size, 'header_bytes': header,
        'formats': {'expert_down_IQ4_NL': 46, 'expert_down_Q2_0': 2, 'edited_output_F16': 48},
        'metadata_description': description,
        'objective': 'Test precision of edited output projections without changing any expert payload or runtime setting.',
        'metric_note': 'Lower weight reconstruction error is a pilot result, not a model quality or speed gain. Independent task scores, paired text prediction and fixed runtime testing decide acceptance.',
        'variant_status_sha256': hashlib.sha256((variants / 'status.json').read_bytes()).hexdigest(),
        'source_revision': status['metadata']['source_revision'],
        'source_layout_gate': '960 BF16 chunks re-quantized to Q8 exactly match preserved control payloads; catches wrong source mapping and double SSM permutation',
    }
    output.write_text(json.dumps(plan, indent=2) + '\n', encoding='utf-8', newline='\n')
    print(json.dumps({'allocation': str(output), 'transformer_bytes': predicted, 'total_bytes': plan['predicted_total_bytes'], 'packed_precision_delta': delta, 'quality_gain_established': False}))


if __name__ == '__main__':
    main()
