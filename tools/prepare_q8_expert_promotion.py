"""Bounded Q8 precision pilot for the control's two Q2 expert-down tensors.

Retain all other weights. Check the BF16 source against original IQ4 error
records and compare reconstruction to both control Q2 and original IQ4.
No weight learning or capability gain is inferred from these diagnostics.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sqlite3
import time

from tools.assemble_refined_trial import CONTROL, ROOT, gguf, np, predicted_first_shard_bytes
from tools.assemble_calibrated import variant_path
from tools.download_ablated_text import REVISION, selected_shards
from tools.prepare_dense_precision import read_header

FOLDER=ROOT/'data/optimization-48h'
POOL=FOLDER/'q8-promotion-variants'
NAMES=('blk.5.ffn_down_exps.weight','blk.13.ffn_down_exps.weight')

def sha(raw): return hashlib.sha256(raw).hexdigest()

def dequantize(packed,qtype):
    if qtype!=gguf.GGMLQuantizationType.Q2_0:
        return gguf.quants.dequantize(packed,qtype)
    blocks=packed.reshape(-1,18)
    scale=blocks[:,:2].copy().view('<f2').astype(np.float32)
    codes=np.stack([(blocks[:,2:]>>shift)&3 for shift in (0,2,4,6)],axis=-1)
    return ((codes.reshape(-1,64).astype(np.float32)-1)*scale).reshape(packed.shape[0],640)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--all',action='store_true')
    args=parser.parse_args()
    started=time.monotonic()
    budget=json.loads((FOLDER/'budget.json').read_text(encoding='utf-8-sig'))
    deadline=datetime.fromisoformat(budget['deadline_utc'].replace('Z','+00:00'))
    def check():
        if datetime.now(timezone.utc)>=deadline: raise TimeoutError('Original optimisation deadline reached')
    check()
    source=ROOT/'data/source'
    source_manifest=json.loads((source/'text-source-manifest.json').read_text(encoding='utf-8'))
    if source_manifest['revision']!=REVISION or not source_manifest['complete']: raise ValueError('Pinned source incomplete')
    selected,_=selected_shards(source)
    mapping=json.loads((source/'model.safetensors.index.json').read_text(encoding='utf-8'))['weight_map']
    reader=gguf.GGUFReader(CONTROL)
    control={tensor.name:tensor for tensor in reader.tensors}
    old=sqlite3.connect(f'file:{(ROOT/"data/variants/state.sqlite").as_posix()}?mode=ro',uri=True)
    old_meta=dict(old.execute('SELECT name,value FROM metadata'))
    if old_meta['source_revision']!=REVISION: raise ValueError('Original variants use another source')
    POOL.mkdir(parents=True,exist_ok=True)
    connection=sqlite3.connect(POOL/'state.sqlite')
    connection.execute('CREATE TABLE IF NOT EXISTS chunks(name TEXT,qtype TEXT,part INTEGER,offset INTEGER,size INTEGER,sha256 TEXT,metrics TEXT,PRIMARY KEY(name,qtype,part))')
    connection.execute('CREATE TABLE IF NOT EXISTS metadata(name TEXT PRIMARY KEY,value TEXT)')
    metadata={'source_revision':REVISION,'source_manifest_sha256':sha((source/'text-source-manifest.json').read_bytes()),
              'source_index_sha256':sha((source/'model.safetensors.index.json').read_bytes()),
              'builder_sha256':sha(Path(__file__).read_bytes()),'experts_per_part':'1',
              'method':'Ordinary per-block Q8_0 from pinned edited BF16, two layers only; no fresh GSQ/RCO or weight learning',
              'control_sha256':'84512ed5aaa14930c56345eaaa88adccddce5a80a700b102c0b9fbfdc955f507',
              'original_variant_state_sha256':sha((ROOT/'data/variants/state.sqlite').read_bytes())}
    for key,value in metadata.items():
        prior=connection.execute('SELECT value FROM metadata WHERE name=?',(key,)).fetchone()
        if prior and prior[0]!=value: raise ValueError(f'Resume metadata changed:{key}')
        connection.execute('INSERT OR IGNORE INTO metadata VALUES (?,?)',(key,value))
    connection.commit()
    experts=range(512) if args.all else (0,1,254,255,510,511)
    for name in NAMES:
        layer=int(name.split('.')[1])
        hf=f'model.language_model.layers.{layer}.mlp.experts.down_proj'
        if hf not in selected: raise ValueError('Source is outside edited tensor manifest')
        filename=mapping[hf]
        if Path(filename).name!=filename: raise ValueError('Unsafe source filename')
        shard=source/filename
        header,payload_offset,header_sha=read_header(shard)
        entry=header[hf]
        if entry['dtype']!='BF16' or entry['shape']!=[512,2560,640] or tuple(reversed(control[name].shape))!=(512,2560,640): raise ValueError('Expert source/control shape mismatch')
        begin,end=entry['data_offsets']
        if end-begin!=512*2560*640*2 or begin<0 or payload_offset+end>shard.stat().st_size or shard.stat().st_size!=source_manifest['files'][filename]['bytes']: raise ValueError('Source bounds or length changed')
        if control[name].tensor_type!=gguf.GGMLQuantizationType.Q2_0: raise ValueError('Only the two Q2 control layers may change')
        source_expert_bytes=2560*640*2
        q2_expert_bytes=2560*640//64*18
        iq4_expert_bytes=2560*640//32*18
        q8_expert_bytes=2560*640//32*34
        output=variant_path(POOL,name,'Q8_0')
        original=variant_path(ROOT/'data/variants',name,'IQ4_NL')
        with shard.open('rb') as handle,original.open('rb') as iq4_handle,output.open('r+b' if output.exists() else 'w+b') as destination:
            if output.stat().st_size not in (0,512*q8_expert_bytes): raise ValueError('Existing Q8 payload size differs')
            destination.truncate(512*q8_expert_bytes)
            for expert in experts:
                check()
                handle.seek(payload_offset+begin+expert*source_expert_bytes)
                raw=handle.read(source_expert_bytes)
                if len(raw)!=source_expert_bytes: raise ValueError('Truncated BF16 expert')
                matrix=(np.frombuffer(raw,dtype='<u2').astype(np.uint32)<<16).view(np.float32).reshape(2560,640)
                if not np.isfinite(matrix).all(): raise ValueError('Non-finite source')
                original_part=expert//2
                old_record=old.execute('SELECT offset,size,sha256,metrics FROM chunks WHERE name=? AND qtype=? AND part=?',(name,'IQ4_NL',original_part)).fetchone()
                if not old_record or old_record[:2]!=(original_part*iq4_expert_bytes*2,iq4_expert_bytes*2): raise ValueError('Original variant index changed')
                iq4_handle.seek(old_record[0])
                parent=iq4_handle.read(old_record[1])
                if sha(parent)!=old_record[2]: raise ValueError('Original IQ4 variant changed')
                iq4=np.frombuffer(parent[(expert%2)*iq4_expert_bytes:(expert%2+1)*iq4_expert_bytes],dtype=np.uint8).reshape(2560,640//32*18)
                q2=control[name].data.reshape(-1)[expert*q2_expert_bytes:(expert+1)*q2_expert_bytes].reshape(2560,640//64*18)
                q8=gguf.quants.quantize(matrix,gguf.GGMLQuantizationType.Q8_0)
                if q8.shape!=(2560,640//32*34): raise ValueError('Q8 packing layout mismatch')
                metrics={'source_sha256':sha(raw),'source_header_sha256':header_sha,'original_iq4_sha256':sha(iq4.tobytes()),'control_q2_sha256':sha(q2.tobytes()),'source_power':float(np.square(matrix,dtype=np.float64).sum())}
                for label,packed,qtype in (('q2',q2,gguf.GGMLQuantizationType.Q2_0),('iq4',iq4,gguf.GGMLQuantizationType.IQ4_NL),('q8',q8,gguf.GGMLQuantizationType.Q8_0)):
                    reconstructed=dequantize(packed,qtype)
                    if not np.isfinite(reconstructed).all(): raise ValueError('Non-finite reconstructed payload')
                    metrics[label+'_sse']=float(np.square(reconstructed-matrix,dtype=np.float64).sum())
                if metrics['q8_sse']>min(metrics['q2_sse'],metrics['iq4_sse']): raise ValueError('Q8 source reconstruction worse than lower precision')
                metrics['selected_sse']=metrics['q8_sse']
                encoded=q8.tobytes()
                record=(expert*q8_expert_bytes,len(encoded),sha(encoded),json.dumps(metrics,sort_keys=True))
                previous=connection.execute('SELECT offset,size,sha256,metrics FROM chunks WHERE name=? AND qtype=? AND part=?',(name,'Q8_0',expert)).fetchone()
                if previous:
                    destination.seek(record[0])
                    if tuple(previous)!=record or sha(destination.read(record[1]))!=record[2]: raise ValueError('Resume source/metrics/payload changed')
                else:
                    destination.seek(record[0]);destination.write(encoded);destination.flush()
                    connection.execute('INSERT INTO chunks VALUES (?,?,?,?,?,?,?)',(name,'Q8_0',expert,*record));connection.commit()
                if expert%2:
                    left=json.loads(connection.execute('SELECT metrics FROM chunks WHERE name=? AND qtype=? AND part=?',(name,'Q8_0',expert-1)).fetchone()[0])
                    prior=json.loads(old_record[3])
                    for new_key,old_key in (('source_power','source_power'),('iq4_sse','selected_sse')):
                        if not math.isclose(left[new_key]+metrics[new_key],prior[old_key],rel_tol=1e-6,abs_tol=1e-9): raise ValueError('BF16 mapping disagrees with original calibrated source/error records')
                if expert in (0,255,511) or expert%64==63: print(f'{name}: saved expert{expert+1}/512',flush=True)
    records=[json.loads(row[0]) for row in connection.execute('SELECT metrics FROM chunks')]
    count=len(records)
    elapsed=time.monotonic()-started
    summary={'status':'complete' if count==1024 else 'pilot-complete','chunks':count,'q8_sse':sum(row['q8_sse'] for row in records),'iq4_sse':sum(row['iq4_sse'] for row in records),'q2_sse':sum(row['q2_sse'] for row in records),'seconds':elapsed,'scope':'Weight reconstruction only; whole-model quality and speed untested','deadline_utc':budget['deadline_utc']}
    (POOL/'status.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
    if count==1024:
        description='Local trial: retained ISTA GSQ-RCO and corrected abliterated control. Only expert-down layers5 and13 promoted Q2_0 toordinary Q8_0 frompinned BF16. Other46experts,98dense andlookup unchanged. No freshGSQ/RCO orweightlearning; fixedquality/runtime tests required.'
        replacements={name:'Q8_0' for name in NAMES}
        target=FOLDER/'allocation-q8-down.json'
        predicted,header_bytes=predicted_first_shard_bytes(reader,replacements,target.with_suffix('.header.bin'),description)
        old_header=min(tensor.data_offset for tensor in reader.tensors)
        expected=CONTROL.stat().st_size+2*(q8_expert_bytes-q2_expert_bytes)*512+header_bytes-old_header
        if expected!=predicted: raise ValueError('Exact serialized Q8 bytes differ')
        lookup=Path(str(CONTROL).replace('-00001-of-00002','-00002-of-00002'))
        plan={'feasible':True,'control':str(CONTROL),'variants':str(POOL),'replacement_formats':replacements,'predicted_transformer_bytes':predicted,'transformer_budget_bytes':predicted,'predicted_total_bytes':predicted+lookup.stat().st_size,'retained_control_transformer_sha256':metadata['control_sha256'],'metadata_description':description,'formats':{'IQ4_NL':46,'Q8_0':2},'metric_note':'Larger near-source precision trial, not a claimed compression/quality/speed improvement'}
        target.write_text(json.dumps(plan,indent=2)+'\n',encoding='utf-8')
        summary['allocation']=str(target);summary['predicted_total_bytes']=plan['predicted_total_bytes']
    connection.close();old.close()
    print(json.dumps(summary),flush=True)

if __name__=='__main__': main()
