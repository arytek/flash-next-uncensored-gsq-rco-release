"""Create fresh training and operator-validation text without touching final holdout."""
from pathlib import Path
import hashlib
import json
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.prepare_calibration import write_excerpts

corpus_path = ROOT / 'data/calibration/builds/qwen3.8-flash-next-moe/calib_train.txt'
text = corpus_path.read_text(encoding='utf-8')
old = json.loads((ROOT / 'data/calibration/selected/manifest.json').read_text())
out = ROOT / 'data/optimization-48h/calibration'
out.mkdir(parents=True, exist_ok=True)
# Every new excerpt is in a gap between old training excerpts. Validation
# uses distinct gaps. The old final holdout (last 10%) remains untouched.
train_starts, validation_starts = [], []
for i, position in enumerate(old['train_starts']):
    train_starts.append(position + 12000)
    if i % 4 == 0: validation_starts.append(position + 50000)

def emit(starts, filename):
    intervals = [(p, p+2800) for p in starts]
    if any(b > len(text)*9//10 for a,b in intervals): raise ValueError('Final holdout overlap')
    all_old = [(p,p+2800) for p in old['train_starts']+old['heldout_starts']]
    if any(a < d and c < b for a,b in intervals for c,d in all_old): raise ValueError('Old excerpt overlap')
    excerpts = [''.join(c if c in '\t\n\r' or ord(c)>=32 else ' ' for c in text[p:p+2800]) for p in starts]
    (out / filename).write_text('\n\n'.join(excerpts)+'\n',encoding='utf-8')
emit(train_starts,'train.txt')
emit(validation_starts,'validation.txt')
assert not any(a < b+2800 and b < a+2800 for a in train_starts for b in validation_starts)
(out/'manifest.json').write_text(json.dumps({'corpus_revision':old['revision'],
    'corpus_sha256':hashlib.sha256(corpus_path.read_bytes()).hexdigest(),
    'train_starts':train_starts,'validation_starts':validation_starts,
    'selection':'128 fresh disjoint 2800-character train excerpts and 32 validation excerpts; native chunks contain 512 tokens; excerpt boundaries need not match token chunks',
    'activation_source':'corrected quantized control; BF16 ablated weights supply per-operator targets'},indent=2)+'\n')
print('Fresh training and validation text prepared; final holdout preserved.')
