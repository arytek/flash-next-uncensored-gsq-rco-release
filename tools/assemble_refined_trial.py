"""Write a private first-shard trial and reuse the unchanged lookup shard.

The corrected control supplies every tensor except specified edited variants.
Tensor ranges and shapes are checked after writing. No publication.
"""
from pathlib import Path
import argparse
import hashlib
import json
import math
import os
import sqlite3
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/'third_party/llama.cpp/gguf-py'))
import gguf
import numpy as np
from tools.assemble_hybrid import copy_metadata
from tools.assemble_calibrated import variant_path
from tools.verify_calibrated import equal_ranges
from tools.verify_release import file_sha256

CONTROL=Path('models/flash-next-uncensored-gsq-rco-models/Qwen3.8-Flash-Next-Abliterated-GSQ-RCO-IQ3_XXS-Calibrated-00001-of-00002.gguf')

def replacement_bytes(shape,fmt):
    if fmt not in {'Q2_0','IQ4_NL','Q4_0','Q8_0','F16'}:
        raise ValueError(f'Unsupported trial format: {fmt}')
    block,block_bytes=gguf.GGML_QUANT_SIZES[gguf.GGMLQuantizationType[fmt]]
    if shape[-1]%block:
        raise ValueError(f'Tensor width incompatible with {fmt}; no fallback')
    return math.prod(shape)//block*block_bytes

def trial_writer(source,output,description=None):
    writer=gguf.GGUFWriter(output,'qwen4exp')
    copy_metadata(writer,source,'IQ3_XXS Refined',
        description or 'Local trial: retained ISTA GSQ-RCO weights and corrected abliterated control tensors. Selected expert-down weights re-quantized from BF16 with signed-scale/native candidates and separate routed operator validation. Targeted adaptation, not full upstream GSQ/RCO.')
    for name,field in source.fields.items():
        if name.startswith('split.'):
            writer.add_key_value(name,field.contents(),field.types[0])
    writer.data_alignment=int(source.alignment)
    return writer

def predicted_first_shard_bytes(source,replacements,preview,description=None):
    writer=trial_writer(source,preview,description)
    packed=0
    for tensor in source.tensors:
        shape=tuple(int(v) for v in reversed(tensor.shape))
        fmt=replacements.get(tensor.name)
        qtype=gguf.GGMLQuantizationType[fmt] if fmt else tensor.tensor_type
        nbytes=replacement_bytes(shape,fmt) if fmt else tensor.n_bytes
        writer.add_tensor_info(tensor.name,shape,np.dtype(np.float32),nbytes,raw_dtype=qtype)
        packed+=gguf.GGUFWriter.ggml_pad(nbytes,writer.data_alignment)
    preview.parent.mkdir(parents=True,exist_ok=True)
    writer.write_header_to_file();writer.write_kv_data_to_file();writer.write_ti_data_to_file()
    header=gguf.GGUFWriter.ggml_pad(writer.fout[0].tell(),writer.data_alignment)
    writer.close()
    return header+packed,header
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--control',type=Path,default=CONTROL)
    p.add_argument('--variants',type=Path,default=ROOT/'data/optimization-48h/variants')
    p.add_argument('--layers',default='0,23,47')
    p.add_argument('--format',default='IQ4_NL',choices=['IQ4_NL','Q4_0','Q2_0'])
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--hash',action='store_true')
    p.add_argument('--allocation',type=Path)
    args=p.parse_args()
    if not str(args.output).endswith('-00001-of-00002.gguf'): p.error('Output must name first of two shards')
    if args.output.resolve()==args.control.resolve(): raise ValueError('Cannot overwrite control')
    if args.output.exists(): raise FileExistsError('Choose another trial path')
    layers=list(range(48)) if args.layers=='all' else list(map(int,args.layers.split(',')))
    plan=json.loads(args.allocation.read_text()) if args.allocation else None
    if plan and not plan.get('feasible'): raise ValueError('Allocation is infeasible')
    if plan and Path(plan['variants']).resolve()!=args.variants.resolve(): raise ValueError('Allocation uses another variant pool')
    replacements=plan['replacement_formats'] if plan else {f'blk.{i}.ffn_down_exps.weight':args.format for i in layers}
    if plan and Path(plan['control']).resolve()!=args.control.resolve(): raise ValueError('Allocation uses another control')
    source=gguf.GGUFReader(args.control)
    names={t.name for t in source.tensors}
    if replacements.keys()-names: raise ValueError('Unknown replacement tensor')
    conn=sqlite3.connect(args.variants/'state.sqlite')
    metadata=dict(conn.execute('SELECT name,value FROM metadata'))
    source_tensors={t.name:t for t in source.tensors}
    for name,fmt in replacements.items():
        tensor=source_tensors[name]
        shape=tuple(int(v) for v in reversed(tensor.shape))
        expected_bytes=replacement_bytes(shape,fmt)
        if fmt=='F16':
            if not name.endswith(('.ssm_out.weight','.attn_output.weight')) or tensor.tensor_type!=gguf.GGMLQuantizationType.Q8_0:
                raise ValueError('F16 is restricted to control Q8 edited output projections')
            if metadata.get('rows_per_part')!='128' or metadata.get('source_revision')!='deb02632504bb214702bc28b0381a93d3112f500':
                raise ValueError('F16 pool protocol or pinned source changed')
            expected_parts=math.ceil(shape[0]/128)
        else:
            if not name.endswith('.ffn_down_exps.weight'):
                raise ValueError('Packed low-bit variants are restricted to expert-down tensors')
            if fmt=='Q8_0' and (name not in {'blk.5.ffn_down_exps.weight','blk.13.ffn_down_exps.weight'} or metadata.get('source_revision')!='deb02632504bb214702bc28b0381a93d3112f500' or metadata.get('experts_per_part')!='1'):
                raise ValueError('Q8 expert promotion is restricted to the pinned two-layer pilot')
            expected_parts=512
        entries=conn.execute('SELECT offset,size,sha256 FROM chunks WHERE name=? AND qtype=? ORDER BY part',(name,fmt)).fetchall()
        if len(entries)!=expected_parts: raise ValueError(f'Incomplete trial pool: {name} {fmt}')
        position=0
        with variant_path(args.variants,name,fmt).open('rb') as f:
            for offset,size,digest in entries:
                if offset!=position or hashlib.sha256(f.read(size)).hexdigest()!=digest: raise ValueError('Variant chunk verification failed')
                position+=size
            if position!=expected_bytes or f.read(1): raise ValueError('Variant byte coverage differs from tensor format')
    conn.close()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    writer=trial_writer(source,args.output,plan.get('metadata_description') if plan else None)
    for tensor in source.tensors:
        fmt=replacements.get(tensor.name)
        if fmt:
            shape=tuple(int(v) for v in reversed(tensor.shape))
            if fmt=='F16':
                payload=np.memmap(variant_path(args.variants,tensor.name,fmt),dtype='<f2',mode='r',shape=shape)
            else:
                block,block_bytes=gguf.GGML_QUANT_SIZES[gguf.GGMLQuantizationType[fmt]]
                payload=np.memmap(variant_path(args.variants,tensor.name,fmt),dtype=np.uint8,mode='r',
                                  shape=(*shape[:-1],shape[-1]//block*block_bytes))
            writer.add_tensor(tensor.name,payload,raw_dtype=gguf.GGMLQuantizationType[fmt])
        else:
            writer.add_tensor(tensor.name,tensor.data,raw_dtype=tensor.tensor_type)
    writer.write_header_to_file();writer.write_kv_data_to_file();writer.write_tensors_to_file(progress=True);writer.close()
    if plan and (args.output.stat().st_size>plan['transformer_budget_bytes'] or args.output.stat().st_size!=plan['predicted_transformer_bytes']):
        raise ValueError('Written model differs from exact byte allocation')
    lookup_source=Path(str(args.control).replace('-00001-of-00002.gguf','-00002-of-00002.gguf'))
    lookup_target=Path(str(args.output).replace('-00001-of-00002.gguf','-00002-of-00002.gguf'))
    if lookup_target.exists(): raise FileExistsError('Lookup target already exists')
    os.link(lookup_source,lookup_target)
    output=gguf.GGUFReader(args.output)
    old_tensors={t.name:t for t in source.tensors}
    if {t.name for t in output.tensors}!=names: raise ValueError('Trial tensor names changed')
    for i,tensor in enumerate(output.tensors):
        old=old_tensors[tensor.name]
        if tuple(tensor.shape)!=tuple(old.shape): raise ValueError('Trial tensor shape changed')
        fmt=replacements.get(tensor.name)
        path=variant_path(args.variants,tensor.name,fmt) if fmt else args.control
        offset=0 if fmt else old.data_offset
        expected_type=gguf.GGMLQuantizationType[fmt] if fmt else old.tensor_type
        if tensor.tensor_type!=expected_type or not equal_ranges(path,offset,args.output,tensor.data_offset,tensor.n_bytes):
            raise ValueError(f'Trial payload mismatch: {tensor.name}')
        if i%100==0: print(f'Verified {i+1}/{len(output.tensors)} trial tensors',flush=True)
    report={'scope':'private development trial, not accepted release','control':str(args.control),
        'output_shards':[str(args.output),str(lookup_target)],'replacement_formats':replacements,
        'verified_transformer_tensors':len(output.tensors),'lookup_shard':'hard link to preserved control; no packed payload changes',
        'bytes':[args.output.stat().st_size,lookup_target.stat().st_size],
        'variant_pool_metadata':dict(sqlite3.connect(args.variants/'state.sqlite').execute('SELECT name,value FROM metadata'))}
    if plan: report['allocation']=plan
    if args.hash: report['sha256']=[file_sha256(args.output),file_sha256(lookup_target)]
    args.output.with_suffix('.trial.json').write_text(json.dumps(report,indent=2)+'\n')
    print('Trial assembled and tensor ranges verified',flush=True)

if __name__=='__main__': main()
