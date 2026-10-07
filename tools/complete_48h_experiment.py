"""Record the completed bounded experiment without selecting a replacement."""
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FOLDER = ROOT / 'data/optimization-48h'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')


def main():
    budget_path, status_path = FOLDER / 'budget.json', FOLDER / 'status.json'
    budget, status = read(budget_path), read(status_path)
    assert budget['started_utc'] == '2026-09-30T18:08:13Z'
    assert budget['deadline_utc'] == '2026-10-02T18:08:13Z'
    cleanup = read(FOLDER / 'final-cleanup-audit.json')
    source_audit = read(FOLDER / 'final-retained-source-audit.json')
    review_audit = read(FOLDER / 'final-q8-review-audit.json')
    docs_sync = read(FOLDER / 'final-model-docs-sync.json')
    for record in (cleanup, source_audit, review_audit, docs_sync):
        assert record['status'] == 'complete'
    assert cleanup['removed_first_shard_bytes_this_run'] == 110707125696
    for filename in ('confirmation-status.json', 'q8-confirmation-status.json'):
        assert read(FOLDER / filename)['status'] == 'skipped-no-improvement'
    model_root = Path(docs_sync['model_directory'])
    for entry in docs_sync['files']:
        for path in (ROOT / entry['file'], model_root / entry['file']):
            assert hashlib.sha256(path.read_bytes()).hexdigest() == entry['sha256']
    assert not list((model_root / 'optimization-48h').glob('*.gguf'))
    # Check local document links only; remote sources were reviewed earlier.
    for document in ROOT.glob('*.md'):
        for target in re.findall(r'(?<!!)\[[^\]]+\]\(([^)]+)\)', document.read_text(encoding='utf-8-sig')):
            if '://' in target or target.startswith('#'):
                continue
            assert (document.parent / target.split('#')[0]).exists(), (document, target)
    q8 = read(FOLDER / 'q8-coding-review/summary.json')
    now = datetime.now(timezone.utc)
    stamp = now.isoformat()
    elapsed = (now - datetime.fromisoformat(budget['started_utc'].replace('Z', '+00:00'))).total_seconds() / 3600
    # Preserve earlier status rather than losing the resumable experiment history.
    snapshot = FOLDER / 'status-before-final-completion.json'
    if not snapshot.exists():
        snapshot.write_bytes(status_path.read_bytes())
    completion = {
        'utc': stamp, 'status': 'completed-bounded-experiment', 'elapsed_window_hours': elapsed,
        'immutable_started_utc': budget['started_utc'], 'immutable_deadline_utc': budget['deadline_utc'],
        'ended_before_deadline': now < datetime.fromisoformat(budget['deadline_utc'].replace('Z', '+00:00')),
        'reason': 'All scheduled comparisons and remaining diagnostics finished; another model build/validation could not meet the six-hour reserve.',
        'candidate_selected': False, 'acceptance_complete': False,
        'decision': 'No tested replacement established an overall improvement over the preserved control.',
        'control': {'total_bytes': 83144446848, 'generation_mean_tokens_per_second': 20.539091,
                    'generation_min_tokens_per_second': 20.4028, 'transformer_sha256': cleanup['control_sha256'],
                    'lookup_sha256': cleanup['lookup_sha256'], 'model_directory': str(model_root),
                    'scope': 'Earlier five short synthetic512 runs, not populated16K or long-context acceptance'},
        'baseline': {'total_bytes': 75839998528, 'generation_mean_tokens_per_second': 14.485683,
                     'scope': 'Untouched local ISTA IQ3_XXS; retained source files, unknown Hub revision'},
        'q8_final': q8, 'coding_task_model_records': 656, 'exploratory_response_records': 80,
        'unresolved': ['Quality margin uncertainty', 'SixGiB availableRAM aim',
                       'Long-context acceptance', 'FullBF16 or mixedBF16 teacher fidelity',
                       'General ablation/refusal benefit', 'Portable reproduction and other-hardware performance'],
        'provenance': 'Inherited ISTA GSQ-RCO plus targeted local edited-weight methods; not full upstreamGSQ/RCO. AtomicChat recipe/corpus, no AtomicChat weights.',
        'cleanup': cleanup, 'source_audit': 'final-retained-source-audit.json',
        'review_audit': 'final-q8-review-audit.json', 'screening_audit': 'final-precision-screening-audit.json',
        'reports': ['OPTIMIZATION-48H.md', 'COMPARISON-REPORT.md', 'CODING-REVIEW.md', 'BEHAVIOR-REVIEW.md'],
        'local_private': True, 'uploads_or_code_publication': False, 'populated16K_test_repeated': False,
        'worker_inventory': 'Final elevated inventory found no project model workers; completed source audit process also exited.',
        'followup': 'Delete local-flash-next-optimisation after this record; no further automatic work needed.',
    }
    write(FOLDER / 'completion.json', completion)
    status.update(stage='Bounded experiment complete: no improved replacement selected; control preserved',
                  updated_utc=stamp, experiment_status='completed-bounded-experiment',
                  acceptance_complete=False, candidate_selected=False, control_preserved=True,
                  candidate_validation_ready='All scheduled comparisons complete; no further workers or model search',
                  next='No automatic optimisation remains. Preserve control and private results. Later owner testing, release or a new research budget needs a separate instruction.',
                  completion='data/optimization-48h/completion.json')
    status['q8_additional_review'].update(status='Complete:164 isolated coding tests and20 manual response checks; no established gain',
                                           coding_results=q8, behavior_report='BEHAVIOR-REVIEW.md')
    status['q8_promotion'].update(status='Complete and unselected: larger model, no established overall control improvement; primary coding point limits versusISTA missed',
                                  model_file_state='Retired after hashes/reader/path checks; source pool and reconstruction records retained')
    status['dense_precision']['trial']['model_file_state'] = 'Retired after hashes/reader/path checks; source pool and reconstruction records retained'
    status['final_cleanup'] = cleanup
    status['coding_executor']['human_eval'] = 'Completed164 tests each forISTA/control/original-IQ4/Q8. Raw111/109/111/103; separateformatting128/141/142/143. No primary improvement or model selection.'
    write(status_path, status)
    budget.update(status='completed-bounded-experiment', completed_utc=stamp,
                  actual_elapsed_window_hours=elapsed, candidate_selected=False)
    write(budget_path, budget)
    print(json.dumps({'status': completion['status'], 'elapsed_window_hours': elapsed,
                      'candidate_selected': False, 'control_preserved': True,
                      'c_bytes_freed': cleanup['removed_first_shard_bytes_this_run']}))


if __name__ == '__main__':
    main()
