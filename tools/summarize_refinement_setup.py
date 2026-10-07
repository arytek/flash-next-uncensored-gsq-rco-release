"""Check staged inputs before full refinement and record their limited scope."""
from datetime import datetime, timezone
from pathlib import Path
import json
import statistics

ROOT = Path(__file__).resolve().parents[1]


def main():
    folder = ROOT / 'data/optimization-48h'
    budget = json.loads((folder / 'budget.json').read_text())
    deadline = datetime.fromisoformat(budget['deadline_utc'].replace('Z', '+00:00'))
    if datetime.now(timezone.utc) >= deadline:
        raise TimeoutError('48-hour deadline reached')
    for stage in ('development', 'extended-calibration', 'format-bench'):
        if json.loads((folder / f'{stage}-status.json').read_text())['status'] != 'complete':
            raise ValueError(f'Incomplete prerequisite: {stage}')
    comparisons = {reference: json.loads((folder / f'development-vs-{reference}.json').read_text())
                   for reference in ('control', 'ista')}
    # A small development trial has no acceptance authority. A statistically
    # established aggregate regression stops expansion for manual inspection.
    for reference, comparison in comparisons.items():
        if comparison['macro']['paired_bootstrap_95_percentage_points'][1] < -2:
            raise ValueError(f'Development regression against {reference} requires review')
    captures = {mode: json.loads((folder / f'capture-{mode}-extended/status.json').read_text())
                for mode in ('train', 'validation')}
    if any(record['status'] != 'complete' for record in captures.values()):
        raise ValueError('Merged capture is incomplete')
    weak_train = [r['layer'] for r in captures['train']['coverage'] if r['under_8']]
    weak_valid = [r['layer'] for r in captures['validation']['coverage'] if r['under_4']]
    timings = {}
    for fmt in ('IQ4_NL', 'Q2_0', 'Q4_0'):
        timings[fmt] = {}
        for layout in ('native', 'preferred'):
            path = ROOT / f'logs/expert-format-{fmt}-{layout}.jsonl'
            rows = [json.loads(line) for line in path.read_text().splitlines()
                    if line.strip().startswith('{')]
            if len(rows) != 5 or sorted(r['group'] for r in rows) != list(range(5)):
                raise ValueError(f'Incomplete benchmark: {fmt}/{layout}')
            values = [r['milliseconds_per_routed_matmul'] for r in rows]
            if any(v <= 0 for v in values):
                raise ValueError('Invalid kernel timing')
            timings[fmt][layout] = {
                'mean_ms': statistics.mean(values), 'min_ms': min(values), 'max_ms': max(values),
                'buffers': sorted({r['buffer'] for r in rows}), 'groups': rows}
    result = {
        'utc': datetime.now(timezone.utc).isoformat(), 'deadline_utc': budget['deadline_utc'],
        'development': comparisons,
        'development_decision': 'Expand operator refinement; no model accepted. Small capability samples remain inconclusive.',
        'coverage': {'weak_training_layers': weak_train, 'weak_validation_layers': weak_valid,
                     'mandatory_iq4_layers': sorted(set(weak_train + weak_valid)),
                     'training_chunks': captures['train']['chunk_count'],
                     'validation_chunks': captures['validation']['chunk_count']},
        'kernel_timings': timings,
        'kernel_scope': 'Private llama.cpp 7fe450e; t12, 512 experts, 8 routes/token. Isolated CPU matmul, not user-engine throughput.',
        'allocation_policy': 'IQ4_NL/Q2_0 search. Q4_0 remains a measured comparison until a runtime and quality case supports including it.',
        'control_preserved': True}
    (folder / 'refinement-setup-review.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k not in ('development', 'kernel_timings')}, indent=2), flush=True)


if __name__ == '__main__':
    main()
