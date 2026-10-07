"""Compare packed reconstruction with actual CPU/CUDA kernels on pilot payloads."""
from pathlib import Path
import json
import sys
import subprocess
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.refine_quantization import decode
import gguf

out = ROOT / 'data/optimization-48h/kernel-check'
out.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(1729)
x = rng.normal(size=(5,640)).astype(np.float32)
input_path = out/'inputs.bin'
x.tofile(input_path)
exe = ROOT/'third_party/llama.cpp/build-ninja-release/bin/rco-kernel-check.exe'
report = []
for qtype in ('IQ4_NL','Q2_0','Q4_0'):
    packed = ROOT/f'data/optimization-48h/initial-pilot/layer0-expert0-{qtype}-signed_initializer.bin'
    values = np.fromfile(packed,dtype=np.uint8)
    scales = values.reshape(-1,18)[:,:2].copy().view('<f2')
    if not np.any(scales < 0): raise ValueError('Pilot must exercise negative scales')
    reference = (decode(values,qtype,(256,640)) @ x.T).T
    results = {}
    for backend in ('cpu','cuda'):
        target = out/f'{qtype}-{backend}.bin'
        subprocess.run([str(exe),str(int(getattr(gguf.GGMLQuantizationType,qtype))),str(packed),
                        '256',str(input_path),'5',str(target),backend],check=True)
        actual = np.fromfile(target,dtype=np.float32).reshape(5,256)
        if not np.isfinite(actual).all(): raise ValueError('Kernel returned nonfinite values')
        relative_rmse = float(np.linalg.norm(actual-reference)/np.linalg.norm(reference))
        # Integer dot kernels also quantize the activation operand. Allow its
        # measured rounding noise, while rejecting packing/layout mismatch.
        if relative_rmse > .015: raise ValueError(f'{qtype}/{backend} mismatch {relative_rmse}')
        results[backend] = relative_rmse
    report.append({'format':qtype,'negative_scale_blocks':int((scales<0).sum()),
                   'relative_rmse_vs_decoded_float_matmul':results,'passed':True})
(out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report),flush=True)
