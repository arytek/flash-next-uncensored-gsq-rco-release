"""Extend the frozen local coding review using its unchanged isolated executor.

Never execute generated code on the host. Raw256 answers remain primary;
dedent-stop is a separate diagnostic. Existing reference results are reused
only after checking their immutable source and execution fingerprints.
"""
from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

import numpy as np

from tools import score_humaneval_isolated as isolated
from tools.compare_eval import FILES, values
from tools.evaluate_local import wilson

ROOT = Path(__file__).resolve().parents[1]
FOLDER = ROOT / 'data/optimization-48h/q8-coding-review'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    isolated.check_time()
    FOLDER.mkdir(parents=True, exist_ok=True)
    plan_path = ROOT / 'data/optimization-48h/q8-additional-review-manifest.json'
    plan = read(plan_path)
    prior_folder = ROOT / 'data/optimization-48h/coding-review'
    prior = read(prior_folder / 'manifest.json')
    executor_path = ROOT / 'tools/score_humaneval_isolated.py'
    assert digest(executor_path) == prior['script_sha256'] == plan['executor_sha256']
    assert isolated.IMAGE == prior['image'] == plan['image']
    assert isolated.LIMITS == prior['limits']
    assert isolated.sha(isolated.RUNNER.encode()) == prior['runner_sha256']
    dataset = ROOT / 'data/eval/selected/humaneval.jsonl'
    assert digest(dataset) == prior['dataset_sha256'] == plan['human_eval_sha256']
    problems = [json.loads(line) for line in dataset.read_text(encoding='utf-8').splitlines()]
    source = ROOT / 'data/eval/results/review-q8-down-code/humaneval.jsonl'
    samples = [json.loads(line) for line in source.read_text(encoding='utf-8').splitlines()]
    by_index = {row['index']: row for row in samples}
    assert len(samples) == 164 and set(by_index) == set(range(164))
    for index, row in by_index.items():
        assert row['set'] == 'humaneval' and row['protocol'] == 'chat-512-v1'
        assert row['row_sha256'] == isolated.sha(json.dumps(problems[index], sort_keys=True).encode())
    for label in ('ista', 'control'):
        assert digest(prior['sources'][label]['path']) == prior['sources'][label]['sha256']
    manifest = {'plan_sha256': digest(plan_path), 'script_sha256': digest(__file__),
                'executor_sha256': prior['script_sha256'], 'image': isolated.IMAGE,
                'limits': isolated.LIMITS, 'runner_sha256': prior['runner_sha256'],
                'response_sha256': digest(source), 'dataset_sha256': digest(dataset),
                'reference_manifest_sha256': digest(prior_folder / 'manifest.json'),
                'reference_results_sha256': digest(prior_folder / 'results.jsonl'),
                'count': 164, 'primary': prior['primary'], 'secondary': prior['secondary'],
                'generation': prior['generation'], 'candidate_selected': False,
                'scope': 'Matching Q8 coding review; does not change frozen three-task confirmation eligibility or establish public leaderboard parity'}
    manifest_path = FOLDER / 'manifest.json'
    if manifest_path.exists():
        assert read(manifest_path) == manifest, 'Execution protocol or sources changed'
    else:
        manifest_path.write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    frozen_hash = isolated.sha(json.dumps(manifest, sort_keys=True).encode())
    assert isolated.cli(['info', '--format', '{{.OSType}}']).strip() == 'linux'
    isolated.cli(['image', 'inspect', isolated.IMAGE])
    verification = isolated.verify_sandbox(problems[0])
    (FOLDER / 'executor-verification.json').write_text(json.dumps(verification, indent=2) + '\n')
    output = FOLDER / 'results.jsonl'
    done = {}
    if output.exists():
        for line in output.read_text().splitlines():
            row = json.loads(line)
            assert row['manifest_sha256'] == frozen_hash and row['index'] not in done
            done[row['index']] = row
    with output.open('a', encoding='utf-8', newline='\n') as stream:
        for index, problem in enumerate(problems):
            isolated.check_time()
            response = by_index[index]['response']
            stopped = isolated.stopped_completion(problem['prompt'], response)
            if index in done:
                saved = done[index]
                assert saved['task_id'] == problem['task_id']
                assert saved['raw']['program_sha256'] == isolated.sha(isolated.program(problem, response).encode())
                assert saved['dedent_stop_diagnostic']['program_sha256'] == isolated.sha(isolated.program(problem, stopped).encode())
                continue
            raw = isolated.run_case(isolated.program(problem, response))
            diagnostic = isolated.run_case(isolated.program(problem, stopped)) if stopped != response else {**raw, 'reused_identical_program': True}
            row = {'model': 'q8-down', 'index': index, 'task_id': problem['task_id'],
                   'raw': raw, 'dedent_stop_diagnostic': diagnostic,
                   'trimmed_characters': len(response) - len(stopped),
                   'manifest_sha256': frozen_hash, 'utc': isolated.utc()}
            stream.write(json.dumps(row, sort_keys=True) + '\n'); stream.flush()
            done[index] = row
            if (index + 1) % 16 == 0 or index == 163:
                print(f'Q8: {index + 1}/164 scored in isolation', flush=True)
    references = {}
    prior_frozen_hash = isolated.sha(json.dumps(prior, sort_keys=True).encode())
    for line in (prior_folder / 'results.jsonl').read_text().splitlines():
        row = json.loads(line)
        assert row['manifest_sha256'] == prior_frozen_hash
        if row['model'] in ('ista', 'control'):
            assert (row['model'], row['index']) not in references
            references[row['model'], row['index']] = row
    result = {'utc': isolated.utc(), 'manifest_sha256': frozen_hash, 'samples': 164,
              'scope': manifest['scope'], 'candidate_selected': False, 'paired_comparisons': {}}
    for mode in ('raw', 'dedent_stop_diagnostic'):
        correct = sum(done[index][mode]['outcome'] == 'passed' for index in range(164))
        result[mode] = {'passed': correct, 'pass_at_1': correct / 164, 'wilson_95': wilson(correct, 164)}
    four_tasks = {}
    for reference, folder_name in (('ista', 'ista-xxs-final'), ('control', 'calibrated-final')):
        paired = {}
        for mode in ('raw', 'dedent_stop_diagnostic'):
            delta = [int(done[index][mode]['outcome'] == 'passed') - int(references[reference, index][mode]['outcome'] == 'passed') for index in range(164)]
            rng = random.Random(20261002)
            draws = sorted(sum(rng.choices(delta, k=164)) * 100 / 164 for _ in range(5000))
            paired[mode] = {'difference_pp': sum(delta) * 100 / 164,
                            'bootstrap_95_pp': [draws[124], draws[4874]],
                            'scope': 'Paired164-task bootstrap; no multiple-comparison correction'}
        result['paired_comparisons'][reference] = paired
        rng = np.random.default_rng(1729)
        deltas = {}
        for task, (filename, field, count) in FILES.items():
            deltas[task] = values(ROOT / 'data/eval/results/rco48-q8-down', filename, field, count) - values(ROOT / 'data/eval/results' / folder_name, filename, field, count)
        deltas['humaneval_raw256'] = np.array([int(done[index]['raw']['outcome'] == 'passed') - int(references[reference, index]['raw']['outcome'] == 'passed') for index in range(164)])
        bootstrap = sum(delta[rng.integers(0, len(delta), size=(5000, len(delta)))].mean(axis=1) for delta in deltas.values()) * 25
        four_tasks[reference] = {'macro_difference_pp': sum(delta.mean() for delta in deltas.values()) * 25,
                                 'paired_bootstrap_95_pp': np.quantile(bootstrap, [.025, .975]).tolist(),
                                 'tasks': {task: {'samples': len(delta), 'difference_pp': float(delta.mean() * 100)} for task, delta in deltas.items()},
                                 'scope': 'Additional equal-weight four-task summary, raw coding only. Does not replace frozen three-task screening or revive skipped confirmation.'}
    result['additional_four_task_comparisons'] = four_tasks
    (FOLDER / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
    (FOLDER / 'result-integrity.json').write_text(json.dumps({'records': len(done), 'all_program_and_source_hashes_match': True, 'manifest_sha256': frozen_hash, 'candidate_selected': False}, indent=2) + '\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
