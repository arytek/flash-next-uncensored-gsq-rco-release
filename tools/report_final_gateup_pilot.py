"""Summarise the complete local compression pilot without accepting a model."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PHASE = ROOT / 'data/optimization-final-gateup'


def main():
    prototype = json.loads((PHASE / 'pilot/summary.json').read_text(encoding='utf-8'))
    full_folder = PHASE / 'full-layer-pilot'
    full_status = json.loads((full_folder / 'status.json').read_text(encoding='utf-8-sig'))
    if full_status['status'] == 'complete':
        complete = json.loads((full_folder / 'summary.json').read_text(encoding='utf-8'))
    elif full_status['status'] == 'stopped-after-completed-layer-review':
        complete = {'results': [json.loads(path.read_text(encoding='utf-8')) for path in full_folder.glob('layer*-IQ*.json')],
                    'control': prototype['control'], 'confirmation_used': False, 'elapsed_seconds': None}
    else:
        raise RuntimeError('Full-layer worker is not in a reviewed terminal state')
    published = json.loads((PHASE / 'published-q2-gateup/summary.json').read_text(encoding='utf-8'))
    if prototype['status'] != 'complete' or published['status'] != 'complete':
        raise RuntimeError('Prototype and published-weight checks must be complete')
    if prototype['confirmation_used'] or complete['confirmation_used'] or published['confirmation_used']:
        raise RuntimeError('Pilot unexpectedly used confirmation')
    if complete['control'] != prototype['control']:
        raise RuntimeError('Pilot controls differ')
    rows = []
    for item in complete['results']:
        rows.append({
            'layer': item['layer'], 'format': item['format'],
            'native_validation_mse_ratio': item['native_error'][1]['output_sse'] / item['control_error'][1]['output_sse'],
            'refined_validation_mse_ratio': item['refined_to_control_mse_ratio'][1],
            'native_training_mse_ratio': item['native_error'][0]['output_sse'] / item['control_error'][0]['output_sse'],
            'refined_training_mse_ratio': item['refined_to_control_mse_ratio'][0],
            'poorly_covered_train_experts': item['poorly_covered_train_experts'],
            'seconds': item['packing_and_scale_seconds'],
            'layer_payload_bytes': sum(value['bytes'] for value in item['packed_tensors'].values()),
        })
    parity = max(value['relative_rmse'] for row in prototype['layers'] for value in row['control_parity'])
    report = {'completed_utc': datetime.now(timezone.utc).isoformat(), 'control': prototype['control'],
        'rows': rows, 'published_GSQ_Q2_rows': published['results'], 'published_source': published['source'],
        'native_sweep_status': full_status, 'maximum_control_parity_relative_rmse': parity,
        'prototype_seconds': prototype['elapsed_seconds'], 'full_layer_seconds': complete['elapsed_seconds'],
        'interpretation': 'MSE ratios above1 mean more mixed-reference block error than control; these are not benchmark scores.',
        'scope': 'Three12-expert prototypes; one complete512-expert nativeIQ2_S layer; three complete512-expert publishedGSQ_Q2_0 layers. Remaining native sweep stopped, not tested.',
        'validation_membership': 'Expert membership screened using training and validation routing coverage; validation target outputs never choose payloads.',
        'reference': 'BF16 gate/up, retained quantized ablated down, control inputs/routing; no full BF16 or downstream model reference.',
        'method': 'Native calibrated IQ packing and fixed-codebook global-scale refinement; not a full upstream GSQ/RCO rerun.',
        'models_assembled': 0, 'candidate_accepted': False, 'confirmation_used': False}
    (PHASE / 'pilot-comparison.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    lines = ['# Final compression pilot', '',
             'No replacement model has been accepted. The preserved model remains 83.14 GB.', '',
             'These checks measure expert-layer reconstruction, not answer quality or generation speed.', '',
             '| Layer | Format | Native validation error / control | Refined validation error / control |',
             '|---|---|---:|---:|']
    for row in rows:
        lines.append(f"| {row['layer']} | {row['format']} | {row['native_validation_mse_ratio']:.3f}× | {row['refined_validation_mse_ratio']:.3f}× |")
    lines += ['', 'Lower is better; 1.000× equals the control’s error against the mixed reference.', '',
              'The remaining native sweep was stopped after the first full layer showed substantial fidelity loss. Unfinished formats and layers have no completed result.', '',
              '## Existing published GSQ Q2_0 weights', '',
              '| Layer | Validation error / control |', '|---|---:|']
    for row in published['results']:
        lines.append(f"| {row['layer']} | {row['mse_ratio'][1]:.3f}× |")
    lines += ['', 'The complete ISTA source checksum matches its pinned release. These payloads were retained exactly; no local GSQ rerun or scale updates were applied.', '',
              f'Reconstructed control outputs matched the captured runtime within {parity * 100:.2f}% relative RMSE across all three layers.', '',
              'Every completed candidate tensor contains all 512 experts at its declared format. In the native check, only 12 covered experts receive optional training-only scale updates. Rare experts receive native calibrated packing; this does not establish adequate calibration coverage.', '',
              'The reference combines BF16 gate/up weights with retained ablated quantized down weights and the control’s inputs and routing. It is not a full BF16 model. The local method uses native IQ codebooks and learns their scales; it is not a full GSQ/RCO rerun.', '',
              'Training and validation text intervals are disjoint. Expert membership uses routing coverage from both sets; validation target outputs never select weight updates. Whole-model benchmarks, memory checks and fresh confirmation are required before accepting any build.', '',
              'All sources, payloads and records stay local. The baseline, control and unchanged lookup shard are preserved.', '']
    (ROOT / 'FINAL-PILOT-RESULTS.md').write_text('\n'.join(lines), encoding='utf-8')
    print(json.dumps({'status': 'reported-unaccepted-pilot', 'rows': rows}, indent=2))


if __name__ == '__main__':
    main()
