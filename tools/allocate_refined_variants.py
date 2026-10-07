"""Allocate layer formats under exact GGUF byte limits using routed output error.

This is a local operator-error allocation proxy, not upstream task-loss RCO.
Weakly covered layers retain IQ4_NL. Full capability tests decide acceptance.
"""
from pathlib import Path
import argparse
import json
import math
import sqlite3
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.assemble_refined_trial import CONTROL,predicted_first_shard_bytes
import gguf

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--control',type=Path,default=CONTROL)
    p.add_argument('--variants',type=Path,required=True)
    p.add_argument('--budget-bytes',type=int,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--high-format',choices=['IQ4_NL','Q4_0','best'],default='IQ4_NL')
    args=p.parse_args()
    source=gguf.GGUFReader(args.control)
    tensors={t.name:t for t in source.tensors}
    connection=sqlite3.connect(args.variants/'state.sqlite')
    summaries={}
    mandatory=[];low={};high={}
    names=[f'blk.{layer}.ffn_down_exps.weight' for layer in range(48)]
    for name in names:
        if tuple(reversed(tensors[name].shape))!=(512,2560,640): raise ValueError('Unexpected expert shape')
        formats=['Q2_0','IQ4_NL']+(['Q4_0'] if args.high_format!='IQ4_NL' else [])
        for fmt in formats:
            rows=connection.execute('SELECT part,offset,size,metrics FROM chunks WHERE name=? AND qtype=? ORDER BY part',(name,fmt)).fetchall()
            expected=2560*640//(64 if fmt=='Q2_0' else 32)*18
            if len(rows)!=512 or any(part!=i or offset!=i*expected or size!=expected for i,(part,offset,size,_) in enumerate(rows)):
                raise ValueError(f'Incomplete format pool: {name}/{fmt}')
            metrics=[json.loads(row[3]) for row in rows]
            seen=sum(m['train_seen'] for m in metrics)
            if seen<=0: raise ValueError('No expert routing observations')
            summaries[(name,fmt)]={'mean_routed_output_error':sum(m['allocation_error']*m['train_seen'] for m in metrics)/seen,
                'under_covered_experts':sum(m['experts_with_under_8_samples'] for m in metrics),
                'train_observations':seen,'methods':{method:sum(m['selected_method']==method for m in metrics) for method in sorted({m['selected_method'] for m in metrics})}}
        weak=summaries[(name,'IQ4_NL')]['under_covered_experts']>0
        if weak: mandatory.append(name)
        low[name]='IQ4_NL' if weak else 'Q2_0'
        candidates=['IQ4_NL'] if weak else (['IQ4_NL','Q4_0'] if args.high_format=='best' else [args.high_format])
        high[name]=min(candidates,key=lambda fmt:summaries[(name,fmt)]['mean_routed_output_error'])
    connection.close()
    preview=args.output.with_suffix('.header.bin')
    predicted,header=predicted_first_shard_bytes(source,low,preview)
    if predicted>args.budget_bytes:
        result={'feasible':False,'budget_bytes':args.budget_bytes,'minimum_conservative_transformer_bytes':predicted,
                'mandatory_iq4_layers':[int(name.split('.')[1]) for name in mandatory],
                'reason':'Coverage constraints exceed requested budget; collect more observations or retain higher precision'}
        args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(result),flush=True);return
    choices=dict(low)
    increment=512*2560*640//64*18
    upgrades=sorted((name for name in names if name not in mandatory),key=lambda name:(
        summaries[(name,'Q2_0')]['mean_routed_output_error']-summaries[(name,high[name])]['mean_routed_output_error']),reverse=True)
    for name in upgrades:
        improvement=summaries[(name,'Q2_0')]['mean_routed_output_error']-summaries[(name,high[name])]['mean_routed_output_error']
        if improvement>0 and predicted+increment<=args.budget_bytes:
            choices[name]=high[name];predicted+=increment
    exact,header=predicted_first_shard_bytes(source,choices,preview)
    if exact!=predicted or exact>args.budget_bytes: raise ValueError('Exact GGUF allocation calculation mismatch')
    plan={'feasible':True,'control':str(args.control),'variants':str(args.variants),'replacement_formats':choices,
        'transformer_budget_bytes':args.budget_bytes,'predicted_transformer_bytes':exact,'header_bytes':header,
        'predicted_total_bytes':exact+28800138432,
        'mandatory_iq4_layers':[int(name.split('.')[1]) for name in mandatory],
        'formats':{fmt:list(choices.values()).count(fmt) for fmt in sorted(set(choices.values()))},
        'objective':'Route-frequency-weighted mean BF16 operator output SSE; diagonal variance fallback for unmeasured slots. Weakly covered layers keep IQ4_NL. This is not whole-model RCO task loss.',
        'layers':{name:{fmt:summaries[(name,fmt)] for fmt in formats} for name in names}}
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(plan,indent=2)+'\n')
    print(json.dumps({k:v for k,v in plan.items() if k not in ('layers','replacement_formats')},indent=2),flush=True)

if __name__=='__main__':main()
