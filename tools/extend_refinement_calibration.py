"""Prepare extra disjoint calibration excerpts to improve rare-expert coverage."""
from pathlib import Path
import hashlib
import json
ROOT=Path(__file__).resolve().parents[1]
folder=ROOT/'data/optimization-48h/calibration'
original=json.loads((ROOT/'data/calibration/selected/manifest.json').read_text())
first=json.loads((folder/'manifest.json').read_text())
source=ROOT/'data/calibration/builds/qwen3.8-flash-next-moe/calib_train.txt'
text=source.read_text(encoding='utf-8')
sets={
    'train-extra':[p+20000 for p in original['train_starts']],
    'validation-extra':[p+60000 for p in original['train_starts']],
}
previous=original['train_starts']+original['heldout_starts']+first['train_starts']+first['validation_starts']
all_new=sets['train-extra']+sets['validation-extra']
for i,a in enumerate(all_new):
    if a+2800>len(text)*9//10: raise ValueError('Final holdout overlap')
    if any(a<b+2800 and b<a+2800 for b in previous+all_new[:i]): raise ValueError('Calibration excerpt overlap')
for name,starts in sets.items():
    clean=[''.join(c if c in '\t\n\r' or ord(c)>=32 else ' ' for c in text[p:p+2800]) for p in starts]
    (folder/(name+'.txt')).write_text('\n\n'.join(clean)+'\n',encoding='utf-8')
manifest={**first,'extra_train_starts':sets['train-extra'],'extra_validation_starts':sets['validation-extra'],
    'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
    'selection':'Additional 128 disjoint training and 128 validation excerpts; native capture is limited to 128 chunks each; original final holdout untouched',
    'reason':'Rare-expert coverage: first capture left 19 training-weak layers and 42 with training-or-validation gaps'}
(folder/'extended-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print('Extra calibration excerpts prepared without modifying existing data.')
