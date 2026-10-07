"""Freeze resource limits for the newly approved local gate/up attempt."""
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FOLDER = ROOT / 'data/optimization-final-gateup'


def main():
    FOLDER.mkdir(parents=True, exist_ok=True)
    budget_path = FOLDER / 'budget.json'
    if budget_path.exists():
        print('Existing final-phase resource limits preserved.')
        return
    now = datetime.now(timezone.utc)
    budget = {
        'started_utc': now.isoformat(),
        'pilot_deadline_utc': (now + timedelta(hours=8)).isoformat(),
        'deadline_utc': (now + timedelta(hours=36)).isoformat(),
        'status': 'active-pilot',
        'ceiling_basis': 'Agent-selected upper bounds for this newly approved attempt; not an extension of the closed 48-hour experiment.',
        'publication': 'Local and private. No uploads, code publication, external messages, or paid compute.',
        'preserve_control': True,
        'pilot_layers': [8, 28, 47],
        'pilot_reference': 'BF16 gate/up combined with retained quantized ablated down weights and captured routing; mixed block reference, not a complete BF16 teacher.',
        'objective': 'Nonlinear routed mixture/block-output fidelity; native packing and bounded learned block-scale candidates. A targeted adaptation, not a full upstream GSQ/RCO rerun.',
        'milestone_total_bytes': [79000000000, 81000000000],
        'stretch_total_bytes': 75839998528,
        'max_trial_builds': 2,
        'max_c_trial_models': 1,
        'c_trial_transformer_ceiling_bytes': 54000000000,
        'minimum_c_free_after_build_bytes': 25 * 1024**3,
        'm_working_storage_ceiling_bytes': 220 * 1024**3,
        'pilot_source_ceiling_bytes': 12 * 1024**3,
        'freeze_expert_down_formats': True,
        'freeze_lookup_payload': True,
        'freeze_runtime': 'Engine11199/86a24a182; t12, ncmoe43, ngl999, lazy on, Q8_0/Q5_1 KV, FA on.',
        'speed_acceptance': 'Every one of five sustained short 512-token generations >=18tokens/s; no repeat of owner populated16K speed test and no long-context claim.',
        'quality_acceptance': 'Fresh paired comparison to untouched ISTA and preserved control; report 2pp aggregate/3pp individual margins with uncertainty. Include raw coding, behavior and memory; no automatic public acceptance.',
        'confirmation_rule': 'One candidate selected only from pilot/development, frozen before new confirmation; no tuning from confirmation answers. Do not revive prior unused conditional confirmations.',
        'reserve_hours': 8,
        'infeasible_pilot_action': 'Stop artifact search, preserve evidence and report the blocker; do not fall back silently to the failed operator-only objective.',
        'deadline_action': 'Stop search/build work and finish bounded integrity checks/reporting; preserve original control and records.',
    }
    budget_path.write_text(json.dumps(budget, indent=2) + '\n', encoding='utf-8')
    status = {'utc': now.isoformat(), 'status': 'pilot-preparation',
              'stage': 'Source, kernel and token-level routed-mixture capture preflight',
              'candidate_selected': False, 'control_preserved': True,
              'new_model_assembled': False, 'pilot_reference_scope': budget['pilot_reference']}
    (FOLDER / 'status.json').write_text(json.dumps(status, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'stage': status['stage'], 'pilot_deadline_utc': budget['pilot_deadline_utc'],
                      'deadline_utc': budget['deadline_utc']}))


if __name__ == '__main__':
    main()
