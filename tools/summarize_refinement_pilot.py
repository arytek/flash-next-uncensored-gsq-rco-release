"""Aggregate expert-output pilots without treating them as capability scores."""
import argparse
import json
import statistics
from pathlib import Path
p=argparse.ArgumentParser()
p.add_argument('--input',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
args=p.parse_args()
reports=[json.loads(path.read_text()) for path in sorted(args.input.glob('*.json'))]
if len(reports)!=9: raise ValueError('Expected nine complete expert reports')
summary={'experts':9,'scope':'BF16 operator target, disjoint routed validation inputs from quantized control; not model capability','formats':{}}
for fmt in ('IQ4_NL','Q2_0','Q4_0'):
    rows=[row for report in reports for row in report['results'] if row['format']==fmt]
    methods=set.intersection(*(set(row['scores']) for row in rows))
    sums={method:sum(row['scores'][method]['output_sse'] for row in rows) for method in methods}
    warm_times=[row['method_seconds']['signed_initializer'] for row in rows][1:]
    selected=sum(row['scores'][row['selected']]['output_sse'] for row in rows)
    summary['formats'][fmt]={'output_sse':sums,'selected_output_sse':selected,
        'selected':[row['selected'] for row in rows],
        'improvement_vs_previous_percent':100*(1-selected/sums['previous']) if 'previous' in sums else None,
        'improvement_vs_native_calibrated_percent':100*(1-selected/sums['native_calibrated']),
        'median_warm_initializer_seconds':statistics.median(warm_times),
        'projected_initializer_hours_24576_experts':statistics.median(warm_times)*24576/3600,
        'peak_gpu_bytes':max(row['gpu_peak_bytes'] for row in rows),
        'individual_previous_regressions':sum(row['scores'][row['selected']]['output_sse']>row['scores']['previous']['output_sse'] for row in rows) if 'previous' in sums else None}
args.output.write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2),flush=True)
