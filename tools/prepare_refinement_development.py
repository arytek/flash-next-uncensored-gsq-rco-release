"""Select small disjoint development tasks; preserve final evaluation samples."""
from pathlib import Path
import hashlib
import json
import random
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.prepare_eval_sets import SOURCES, write_jsonl
import pyarrow.parquet as parquet

rng = random.Random(20261001)
out = ROOT / 'data/optimization-48h/development'
out.mkdir(parents=True,exist_ok=True)
manifest = {'seed':20261001,'purpose':'development only; disjoint from fixed final samples','sets':{}}
for name,count in [('mmlu',64),('gsm8k',64),('ifeval',32)]:
    old = [json.loads(line) for line in (ROOT/f'data/eval/selected/{name}.jsonl').read_text(encoding='utf-8').splitlines()]
    old_ids = {json.dumps(row,sort_keys=True) for row in old}
    if name == 'ifeval':
        path = ROOT/'data/eval/ifeval/ifeval_input_data.jsonl'
        rows = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]
    else:
        path = ROOT/SOURCES[name][2]
        rows = parquet.read_table(path).to_pylist()
    available = [row for row in rows if json.dumps(row,sort_keys=True) not in old_ids]
    selected = rng.sample(available,count)
    target = out/f'{name}.jsonl'
    write_jsonl(target,selected)
    manifest['sets'][name] = {'rows':count,'sha256':hashlib.sha256(target.read_bytes()).hexdigest(),
                             'source_sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
(out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print('Separate development tasks prepared.')
