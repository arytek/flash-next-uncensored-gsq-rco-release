"""Merge routed reservoirs uniformly and sum raw calibration statistics.

The hypergeometric merge preserves a uniform reservoir of fixed size while
retaining total observations. Training and validation are merged separately.
"""
from pathlib import Path
import hashlib
import json
import struct
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'third_party/llama.cpp/gguf-py'))
import gguf

def reservoir(path):
    with path.open('rb') as file:
        if file.read(8)!=b'RCOACT01': raise ValueError('Invalid reservoir magic')
        experts,width,cap,reserved=struct.unpack('<4I',file.read(16))
        if (experts,width,reserved)!=(512,640,0): raise ValueError('Invalid reservoir shape')
        counts=np.frombuffer(file.read(experts*8),dtype='<u8').copy()
        data=np.frombuffer(file.read(),dtype='<f2').copy().reshape(experts,cap,width)
    if not np.isfinite(data).all(): raise ValueError('Nonfinite captured vector')
    return counts,data,cap

def main():
    root=ROOT/'data/optimization-48h'
    for mode in ('train','validation'):
        old=root/('capture-'+mode);new=root/('capture-'+mode+'-extra');out=root/('capture-'+mode+'-extended')
        if (out/'status.json').exists() and json.loads((out/'status.json').read_text())['status']=='complete': continue
        for folder in (old,new):
            if json.loads((folder/'status.json').read_text())['status']!='complete': raise ValueError('Incomplete source capture')
        out.mkdir(parents=True,exist_ok=True)
        readers=[gguf.GGUFReader(folder/'imatrix.gguf') for folder in (old,new)]
        tensors=[{t.name:t for t in reader.tensors} for reader in readers]
        writer=gguf.GGUFWriter(out/'imatrix.gguf','imatrix')
        writer.remove_key('general.architecture');writer.add_string('general.type','imatrix')
        chunks=sum(int(reader.get_field('imatrix.chunk_count').contents()) for reader in readers)
        writer.add_uint32('imatrix.chunk_count',chunks);writer.add_uint32('imatrix.chunk_size',512)
        writer.add_key_value('imatrix.datasets',[str(folder/'imatrix.gguf') for folder in (old,new)],gguf.GGUFValueType.ARRAY,gguf.GGUFValueType.STRING)
        coverage=[];hashes={}
        for layer in range(48):
            name=f'blk.{layer}.ffn_down_exps.weight'
            a,xa,ca=reservoir(old/(name+'.act'));b,xb,cb=reservoir(new/(name+'.act'))
            cap=min(ca,cb);counts=a+b
            merged=np.zeros((512,cap,640),dtype='<f2')
            seed=int.from_bytes(hashlib.sha256((mode+name).encode()).digest()[:8],'little')
            rng=np.random.default_rng(seed)
            for ex in range(512):
                n=min(cap,int(counts[ex]))
                if n==0: continue
                na=int(rng.hypergeometric(int(a[ex]),int(b[ex]),n));nb=n-na
                ai=rng.choice(min(ca,int(a[ex])),na,replace=False)
                bi=rng.choice(min(cb,int(b[ex])),nb,replace=False)
                vectors=np.concatenate((xa[ex,ai],xb[ex,bi]),axis=0)
                merged[ex,:n]=vectors[rng.permutation(n)]
            path=out/(name+'.act')
            with path.open('wb') as file:
                file.write(b'RCOACT01');file.write(struct.pack('<4I',512,640,cap,0))
                file.write(counts.astype('<u8').tobytes());file.write(merged.tobytes())
            hashes[name]=hashlib.sha256(path.read_bytes()).hexdigest()
            for suffix in ('.in_sum2','.counts'):
                values=sum(t[name+suffix].data.astype(np.float64) for t in tensors).astype(np.float32)
                if not np.isfinite(values).all(): raise ValueError('Nonfinite merged imatrix statistics')
                if suffix=='.counts' and not np.array_equal(values.reshape(-1),counts.astype(np.float32)):
                    raise ValueError('Reservoir observation counts differ from imatrix counts')
                writer.add_tensor(name+suffix,values)
            coverage.append({'layer':layer,'under_8' if mode=='train' else 'under_4':int((counts<(8 if mode=='train' else 4)).sum())})
        writer.write_header_to_file();writer.write_kv_data_to_file();writer.write_tensors_to_file();writer.close()
        check=gguf.GGUFReader(out/'imatrix.gguf')
        if len(check.tensors)!=96 or check.get_field('imatrix.chunk_count').contents()!=chunks: raise ValueError('Merged imatrix verification failed')
        (out/'status.json').write_text(json.dumps({'status':'complete','mode':mode,'chunk_count':chunks,
            'inputs':[str(old),str(new)],'cap':cap,'merge':'Uniform hypergeometric reservoir merge; raw input-square sums and observation counts added',
            'sha256':hashes,'coverage':coverage},indent=2)+'\n')
        print(f'{mode} captures merged: {chunks} chunks',flush=True)

if __name__=='__main__': main()
