"""Resumable BF16 expert pool with hard-packed operator-validation selection.

Retains old payloads when they win. Records per-expert routing coverage,
method choice, checksums and an output-error allocation proxy. No uploads.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
import sqlite3
import sys
import time
from pathlib import Path
import numpy as np
import torch
from safetensors import safe_open

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.refine_quantization import native_quant, refine, score, read_inputs, BLOCK_SIZES
from tools.quantize_edited import read_imatrix, checked_source, source_name, REPO_REVISION

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for data in iter(lambda:f.read(4*1024*1024),b''): h.update(data)
    return h.hexdigest()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--layers',default='0,23,47')
    p.add_argument('--formats',nargs='+',default=['IQ4_NL','Q2_0','Q4_0'])
    p.add_argument('--source',type=Path,default=ROOT/'data/source')
    p.add_argument('--output',type=Path,default=ROOT/'data/optimization-48h/variants')
    p.add_argument('--steps',type=int,default=1)
    p.add_argument('--budget',type=Path,default=ROOT/'data/optimization-48h/budget.json')
    p.add_argument('--train-inputs',type=Path,default=ROOT/'data/optimization-48h/capture-train')
    p.add_argument('--validation-inputs',type=Path,default=ROOT/'data/optimization-48h/capture-validation')
    p.add_argument('--calibration-manifest',type=Path,default=ROOT/'data/optimization-48h/calibration/manifest.json')
    p.add_argument('--prior-variants',type=Path)
    args=p.parse_args()
    if any(fmt not in ('IQ4_NL','Q2_0','Q4_0') for fmt in args.formats): p.error('Unsupported expert format')
    layers=list(range(48)) if args.layers=='all' else list(map(int,args.layers.split(',')))
    if any(layer not in range(48) for layer in layers): p.error('Invalid layer')
    budget=json.loads(args.budget.read_text())
    deadline=datetime.fromisoformat(budget['deadline_utc'].replace('Z','+00:00'))
    train=args.train_inputs
    valid=args.validation_inputs
    for folder in (train,valid):
        if json.loads((folder/'status.json').read_text())['status']!='complete': raise ValueError('Capture is incomplete')
    imatrix,nchunks=read_imatrix(train/'imatrix.gguf')
    names={f'blk.{layer}.ffn_down_exps.weight' for layer in layers}
    source_files=checked_source(args.source,names)
    args.output.mkdir(parents=True,exist_ok=True)
    connection=sqlite3.connect(args.output/'state.sqlite')
    connection.execute('CREATE TABLE IF NOT EXISTS chunks(name TEXT,qtype TEXT,part INTEGER,offset INTEGER,size INTEGER,sha256 TEXT,metrics TEXT,PRIMARY KEY(name,qtype,part))')
    connection.execute('CREATE TABLE IF NOT EXISTS metadata(name TEXT PRIMARY KEY,value TEXT)')
    version={'source_revision':REPO_REVISION,'method':'signed clipping grid + weighted LS; optional one-step Gumbel; hard-packed per-expert held-out selection v1',
        'steps':str(args.steps),'imatrix_sha256':sha(train/'imatrix.gguf'),
        'refinement_code_sha256':sha(ROOT/'tools/refine_quantization.py'),
        'calibration_manifest_sha256':sha(args.calibration_manifest),
        'old_variant_state_sha256':sha(ROOT/'data/variants/state.sqlite'),
        'ssm_layout':'retained dense tensors use corrected grouped-to-tiled V heads v1',
        'deadline_utc':budget['deadline_utc']}
    if args.prior_variants: version['development_variant_state_sha256']=sha(args.prior_variants/'state.sqlite')
    for key,value in version.items():
        old=connection.execute('SELECT value FROM metadata WHERE name=?',(key,)).fetchone()
        if old and old[0]!=value: raise ValueError(f'Resume provenance mismatch: {key}')
        connection.execute('INSERT OR REPLACE INTO metadata VALUES (?,?)',(key,value))
    connection.commit()
    torch.set_num_threads(8)
    started=time.monotonic()
    completed=0
    for layer in layers:
        name=f'blk.{layer}.ffn_down_exps.weight'
        # These hashes bind the actual routed inputs, not only corpus text.
        for label,folder in [('train',train),('validation',valid)]:
            key=f'{label}_activation_sha256:{name}'
            value=sha(folder/(name+'.act'))
            old=connection.execute('SELECT value FROM metadata WHERE name=?',(key,)).fetchone()
            if old and old[0]!=value: raise ValueError('Captured input changed during resume')
            connection.execute('INSERT OR REPLACE INTO metadata VALUES (?,?)',(key,value))
        connection.commit()
        with safe_open(args.source/source_files[name],framework='pt',device='cpu') as handle:
            sliced=handle.get_slice(source_name(name))
            if sliced.get_shape()!=[512,2560,640]: raise ValueError('Unexpected fused BF16 shape')
            files={}
            for fmt in args.formats:
                path=args.output/f'{name.replace(".","_")}.{fmt}.bin'
                f=path.open('r+b' if path.exists() else 'w+b')
                f.truncate(512*2560*640//BLOCK_SIZES[fmt]*18)
                files[fmt]=f
            try:
                for expert in range(512):
                    if datetime.now(timezone.utc)>=deadline: raise TimeoutError('48-hour deadline reached; build is resumable')
                    pending=[]
                    for fmt in args.formats:
                        size=2560*640//BLOCK_SIZES[fmt]*18
                        old=connection.execute('SELECT sha256 FROM chunks WHERE name=? AND qtype=? AND part=?',(name,fmt,expert)).fetchone()
                        f=files[fmt];f.seek(expert*size)
                        if old and hashlib.sha256(f.read(size)).hexdigest()==old[0]: continue
                        pending.append(fmt)
                    if not pending: continue
                    weight=sliced[expert].float().contiguous()
                    if not torch.isfinite(weight).all(): raise ValueError('Nonfinite BF16 source')
                    values,counts=imatrix[name]
                    factor=np.nan_to_num(values[expert]/max(float(counts[expert]),1),nan=1.,posinf=100.,neginf=1.)
                    if counts[expert]==0: factor=np.ones(640,dtype=np.float32)
                    importance=torch.from_numpy(np.tile(np.clip(factor,1e-8,100.),(2560,1)))
                    train_x,train_seen=read_inputs(train,name,expert)
                    valid_x,valid_seen=read_inputs(valid,name,expert)
                    adequate=len(train_x)>=8 and len(valid_x)>=4
                    for fmt in pending:
                        seed=1729+layer*512+expert
                        torch.manual_seed(seed);torch.cuda.reset_peak_memory_stats()
                        tick=time.monotonic()
                        choices={'native_calibrated':native_quant(weight,fmt,importance)}
                        if fmt in ('IQ4_NL','Q2_0'):
                            old_path=ROOT/f'data/variants/{name.replace(".","_")}.{fmt}.bin'
                            size=2560*640//BLOCK_SIZES[fmt]*18
                            with old_path.open('rb') as old:
                                old.seek(expert*size)
                                choices['previous']=np.frombuffer(old.read(size),dtype=np.uint8).copy()
                        if args.prior_variants:
                            prior_path=args.prior_variants/f'{name.replace(".","_")}.{fmt}.bin'
                            if prior_path.exists():
                                prior=sqlite3.connect(args.prior_variants/'state.sqlite')
                                saved=prior.execute('SELECT offset,size,sha256 FROM chunks WHERE name=? AND qtype=? AND part=?',(name,fmt,expert)).fetchone()
                                prior.close()
                                if saved:
                                    with prior_path.open('rb') as file:
                                        file.seek(saved[0]);payload=np.frombuffer(file.read(saved[1]),dtype=np.uint8).copy()
                                    if hashlib.sha256(payload).hexdigest()!=saved[2]: raise ValueError('Development prior payload changed')
                                    choices['previous_development']=payload
                        choices.update(refine(weight,fmt,args.steps if adequate else 0,importance,train_x))
                        scores={method:score(weight,blob,fmt,importance,valid_x if adequate else None) for method,blob in choices.items()}
                        criterion='output_sse' if adequate else 'weighted_sse'
                        # Poorly covered IQ4_NL experts keep the known control.
                        selected=('previous' if not adequate and fmt=='IQ4_NL' else min(scores,key=lambda method:scores[method][criterion]))
                        metrics={**scores[selected],'selected_method':selected,'method_scores':scores,
                            'train_seen':train_seen,'validation_seen':valid_seen,'train_vectors':len(train_x),'validation_vectors':len(valid_x),
                            'experts_with_under_8_samples':int(not adequate),'experts_with_no_calibration':int(train_seen==0),
                            'allocation_error':scores[selected]['output_sse']/len(valid_x) if adequate else scores[selected]['weighted_sse'],
                            'allocation_reference':'mean per-input BF16 operator output SSE' if adequate else 'diagonal input-variance fallback; conservative IQ4 required',
                            'seed':seed,'seconds':time.monotonic()-tick,'gpu_peak_bytes':torch.cuda.max_memory_allocated()}
                        blob=choices[selected]
                        digest=hashlib.sha256(blob).hexdigest();size=blob.size
                        f=files[fmt];f.seek(expert*size);f.write(blob.tobytes());f.flush()
                        connection.execute('INSERT OR REPLACE INTO chunks VALUES (?,?,?,?,?,?,?)',(name,fmt,expert,expert*size,size,digest,json.dumps(metrics,sort_keys=True)))
                        connection.commit();completed+=1
                    if expert%64==0 or expert==511:
                        status={'stage':'building expert variants','layer':layer,'expert':expert,'completed_this_run':completed,
                            'elapsed_seconds':time.monotonic()-started,'utc':datetime.now(timezone.utc).isoformat(),'deadline_utc':budget['deadline_utc']}
                        (args.output/'progress.json').write_text(json.dumps(status,indent=2)+'\n')
                        print(json.dumps(status),flush=True)
            finally:
                for file in files.values(): file.close()
    connection.close()
    print('Requested variant pool complete',flush=True)

if __name__=='__main__': main()
