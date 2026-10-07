"""Freeze one Q8 candidate confirmation, excluding every earlier selected set."""
from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import pyarrow.parquet as parquet
from tools.prepare_confirmation_sets import ROOT, identity, rows_at, sha

def main():
    folder=ROOT/'data/optimization-48h/confirmation-q8'
    source_manifest_path=ROOT/'data/eval/selected/manifest.json'
    prior=json.loads(source_manifest_path.read_text(encoding='utf-8'))
    rng=random.Random(2026100210)
    records={};exclusions={}
    folder.mkdir(parents=True,exist_ok=True)
    for name in ('mmlu','gsm8k','ifeval'):
        excluded=set();fingerprints={}
        for prefix in ('data/eval/selected','data/optimization-48h/development','data/optimization-48h/confirmation'):
            path=ROOT/prefix/f'{name}.jsonl'
            manifest=json.loads((path.parent/'manifest.json').read_text(encoding='utf-8'))
            if sha(path)!=manifest['sets'][name]['sha256']: raise ValueError('Earlier frozen questions changed')
            excluded.update(identity(row) for row in rows_at(path))
            fingerprints[prefix]=sha(path)
        info=prior['sources'][name];source=ROOT/info['path']
        if sha(source)!=info['sha256']: raise ValueError('Pinned source changed')
        raw=rows_at(source) if name=='ifeval' else parquet.read_table(source).to_pylist()
        available={}
        for row in raw:
            key=identity(row)
            if not key: raise ValueError('Question has no identity')
            if key not in excluded: available.setdefault(key,row)
        if name=='mmlu':
            subjects=defaultdict(list)
            for row in available.values(): subjects[row['subject']].append(row)
            if len(subjects)!=57 or any(len(group)<4 for group in subjects.values()): raise ValueError('Fresh MMLU subject coverage infeasible')
            chosen=[row for subject in sorted(subjects) for row in rng.sample(subjects[subject],4)]
            rng.shuffle(chosen)
        else: chosen=rng.sample(list(available.values()),128 if name=='gsm8k' else 64)
        keys={identity(row) for row in chosen}
        if len(keys)!=len(chosen) or keys&excluded: raise ValueError('Fresh set overlaps earlier questions')
        path=folder/f'{name}.jsonl'
        encoded=''.join(json.dumps(row,ensure_ascii=False,sort_keys=True)+'\n' for row in chosen).encode('utf-8')
        if path.exists() and path.read_bytes()!=encoded: raise ValueError('Frozen Q8 confirmation changed')
        path.write_bytes(encoded)
        records[name]={'rows':len(chosen),'sha256':sha(path),'path':str(path)}
        exclusions[name]={'prior_set_sha256':fingerprints,'excluded_identities':len(excluded),'overlap':0}
    manifest={'frozen_utc':datetime.now(timezone.utc).isoformat(),'selection_seed':2026100210,
              'candidate':'rco48-q8-down','counts':[228,128,64],'sets':records,'sources':prior['sources'],
              'exclusion':exclusions,'builder_sha256':sha(Path(__file__)),
              'candidate_rule':'Only the two-layer Q8 promotion trial, after original fixed-task aggregate improves over control and meets ISTA aggregate>=-2pp/everytask>=-3pp/five fixedshort speeds>=18. No alternative candidate tuning from answers.',
              'reserve':'At least five hours left in original budget; enforce2026-10-02 18:08:13UTC deadline',
              'scope':'Fresh to local experiment, unknown pretraining contamination; no weight/format tuning from these answers. Smaller samples widen uncertainty. Old unused dense-only confirmation preserved.'}
    path=folder/'manifest.json'
    if path.exists():
        previous=json.loads(path.read_text(encoding='utf-8'));manifest['frozen_utc']=previous['frozen_utc']
        if manifest!=previous: raise ValueError('Frozen confirmation protocol changed')
    path.write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps({'manifest':str(path),'counts':manifest['counts'],'overlap_with_any_prior':0,'answers_used_for_tuning':False}))

if __name__=='__main__': main()
