"""Freeze a second local capability sample before the last candidate finishes.

These rows are disjoint from the earlier final and development questions.
Run at most one chosen candidate on them; do not use answers to tune weights.
"""
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random

import pyarrow.parquet as parquet

ROOT = Path(__file__).resolve().parents[1]
FOLDER = ROOT / 'data/optimization-48h/confirmation'
SEED = 2026100204


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def identity(row):
    return ' '.join(row.get('question', row.get('prompt', '')).split())


def rows_at(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]


def main():
    prior_path = ROOT / 'data/eval/selected/manifest.json'
    development_path = ROOT / 'data/optimization-48h/development/manifest.json'
    prior = json.loads(prior_path.read_text(encoding='utf-8'))
    development = json.loads(development_path.read_text(encoding='utf-8'))
    rng = random.Random(SEED)
    selected = {}
    sources = {}
    exclusion = {}
    for name in ('mmlu', 'gsm8k', 'ifeval'):
        earlier = ROOT / f'data/eval/selected/{name}.jsonl'
        dev = ROOT / f'data/optimization-48h/development/{name}.jsonl'
        if sha(earlier) != prior['sets'][name]['sha256'] or sha(dev) != development['sets'][name]['sha256']:
            raise ValueError('Earlier frozen samples changed')
        source_info = prior['sources'][name]
        source = ROOT / source_info['path']
        if sha(source) != source_info['sha256']:
            raise ValueError('Pinned benchmark source changed')
        excluded = {identity(row) for path in (earlier, dev) for row in rows_at(path)}
        raw = rows_at(source) if name == 'ifeval' else parquet.read_table(source).to_pylist()
        # Collapse repeated question text before deterministic selection.
        available = {}
        for row in raw:
            key = identity(row)
            if not key:
                raise ValueError('Benchmark question has no identity')
            if key not in excluded:
                available.setdefault(key, row)
        pool = list(available.values())
        if name == 'mmlu':
            subjects = defaultdict(list)
            for row in pool:
                subjects[row['subject']].append(row)
            if len(subjects) != 57 or any(len(group) < 4 for group in subjects.values()):
                raise ValueError('Cannot preserve all 57 MMLU subjects')
            chosen = [row for subject in sorted(subjects) for row in rng.sample(subjects[subject], 4)]
            rng.shuffle(chosen)
        else:
            chosen = rng.sample(pool, 128 if name == 'gsm8k' else 64)
        keys = {identity(row) for row in chosen}
        if len(keys) != len(chosen) or keys & excluded:
            raise ValueError('Confirmation duplicates or overlaps prior questions')
        selected[name] = chosen
        sources[name] = source_info
        exclusion[name] = {'final_sha256': sha(earlier), 'development_sha256': sha(dev),
                           'excluded_question_identities': len(excluded), 'overlap': 0}
    FOLDER.mkdir(parents=True, exist_ok=True)
    records = {}
    for name, rows in selected.items():
        path = FOLDER / f'{name}.jsonl'
        data = ''.join(json.dumps(row, ensure_ascii=False, sort_keys=True) + '\n' for row in rows).encode('utf-8')
        if path.exists() and path.read_bytes() != data:
            raise ValueError('Frozen confirmation samples changed; preserve old records')
        path.write_bytes(data)
        records[name] = {'rows': len(rows), 'sha256': sha(path), 'path': str(path)}
    manifest_path = FOLDER / 'manifest.json'
    manifest = {
        'frozen_utc': datetime.now(timezone.utc).isoformat(), 'selection_seed': SEED,
        'purpose': 'One independent local confirmation after candidate screening; no weight/format tuning from these responses',
        'sets': records, 'sources': sources, 'exclusion': exclusion,
        'prior_manifest_sha256': sha(prior_path), 'development_manifest_sha256': sha(development_path),
        'counts': [228, 128, 64], 'mmlu_coverage': 'Four fresh questions per each of 57 subjects',
        'candidate_rule': 'Only the frozen dense-F16 trial may be confirmed, after its original fixed-sample scores improve the aggregate versus control, meet all ISTA measured point limits, and preserve all five fixed short speed runs >=18 tokens/s. No alternative candidate selection using these responses.',
        'runtime': 'Same engine86a24a182/build11199 and fixed t12/ncmoe43/Q8_0-Q5_1 settings; no populated16K speed repeat',
        'scope': 'Fresh to this local experiment; benchmark/pretraining contamination is unknown. Smaller samples give wider uncertainty; do not claim that a point estimate passes a confidence gate.',
        'reserve': 'Run only with at least five hours left in the immutable budget; enforce original deadline and preserve partial records if interrupted',
    }
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text(encoding='utf-8'))
        manifest['frozen_utc'] = previous['frozen_utc']
        if manifest != previous:
            raise ValueError('Frozen confirmation protocol changed')
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8', newline='\n')
    print(json.dumps({'counts': manifest['counts'], 'overlap_with_prior_questions': 0,
                      'models_run': 0, 'answers_used_for_tuning': False, 'manifest': str(manifest_path)}))


if __name__ == '__main__':
    main()
