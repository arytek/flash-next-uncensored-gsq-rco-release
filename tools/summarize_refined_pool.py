"""Summarize packed operator comparisons without claiming capability gains."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variants', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    connection = sqlite3.connect(f'{(args.variants / "state.sqlite").resolve().as_uri()}?mode=ro', uri=True)
    totals = {}
    layers = []
    for layer in range(48):
        name = f'blk.{layer}.ffn_down_exps.weight'
        summary = {'layer': layer, 'formats': {}}
        for fmt in ('IQ4_NL', 'Q2_0', 'Q4_0'):
            rows = connection.execute(
                'SELECT part,offset,size,metrics FROM chunks WHERE name=? AND qtype=? ORDER BY part',
                (name, fmt)).fetchall()
            size = 2560 * 640 // (64 if fmt == 'Q2_0' else 32) * 18
            if len(rows) != 512 or any(part != i or offset != i * size or actual != size
                                      for i, (part, offset, actual, _) in enumerate(rows)):
                raise ValueError(f'Incomplete packed layer: {name}/{fmt}')
            path = args.variants / f'{name.replace(".", "_")}.{fmt}.bin'
            if path.stat().st_size != 512 * size:
                raise ValueError(f'Incorrect packed file length: {path}')
            record = {'experts': 512, 'adequate_experts': 0, 'weak_experts': 0,
                      'methods': {}, 'selected_route_weighted_sse': 0.0,
                      'native_route_weighted_sse': 0.0, 'previous_route_weighted_sse': 0.0,
                      'observations_in_operator_comparison': 0, 'previous_compared_experts': 0,
                      'previous_regression_experts': 0}
            for _, _, _, raw in rows:
                metrics = json.loads(raw)
                method = metrics['selected_method']
                record['methods'][method] = record['methods'].get(method, 0) + 1
                if metrics['experts_with_under_8_samples']:
                    record['weak_experts'] += 1
                    continue
                record['adequate_experts'] += 1
                factor = metrics['train_seen'] / metrics['validation_vectors']
                scores = metrics['method_scores']
                record['observations_in_operator_comparison'] += metrics['train_seen']
                record['selected_route_weighted_sse'] += scores[method]['output_sse'] * factor
                record['native_route_weighted_sse'] += scores['native_calibrated']['output_sse'] * factor
                if 'previous' in scores:
                    record['previous_compared_experts'] += 1
                    record['previous_route_weighted_sse'] += scores['previous']['output_sse'] * factor
                    if scores[method]['output_sse'] > scores['previous']['output_sse'] * (1 + 1e-7):
                        record['previous_regression_experts'] += 1
            summary['formats'][fmt] = record
            aggregate = totals.setdefault(fmt, {key: ({} if key == 'methods' else 0)
                                              for key in record})
            for key, value in record.items():
                if key == 'methods':
                    for method, count in value.items():
                        aggregate[key][method] = aggregate[key].get(method, 0) + count
                else:
                    aggregate[key] += value
        layers.append(summary)
    connection.close()
    for record in totals.values():
        selected = record['selected_route_weighted_sse']
        for label in ('native', 'previous'):
            reference = record[f'{label}_route_weighted_sse']
            record[f'reduction_vs_{label}_percent'] = (100 * (1 - selected / reference)
                                                       if reference > 0 else None)
    result = {
        'utc': datetime.now(timezone.utc).isoformat(), 'variants': str(args.variants),
        'scope': 'BF16 expert-down operator output errors, weighted by training route frequency. '
                 'Same operator validation inputs selected the payloads; these are selection metrics. '
                 'Whole-model capability and text-prediction tests remain independent acceptance checks. '
                 'Weak experts excluded from operator error totals; counts and methods retained.',
        'integrity_scope': 'Checks complete row ranges and file lengths; per-payload hashes are verified during assembly.',
        'totals': totals, 'layers': layers}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'scope': result['scope'], 'totals': totals}, indent=2), flush=True)


if __name__ == '__main__':
    main()
