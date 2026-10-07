"""Prepare an ordinary calibrated IQ4 comparison, retaining balanced Q2 weights.

Only IQ4 payload choice changes versus the balanced trial. Weakly observed
experts retain their verified prior payload. No model assembly or uploads.
"""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sqlite3
import sys
import time
import numpy as np
import torch
from safetensors import safe_open

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.quantize_edited import checked_source, read_imatrix, source_name, REPO_REVISION
from tools.refine_quantization import native_quant


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as file:
        for chunk in iter(lambda: file.read(4 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    folder = ROOT / 'data/optimization-48h'
    deadline = datetime.fromisoformat(json.loads((folder / 'budget.json').read_text())['deadline_utc'].replace('Z', '+00:00'))
    if datetime.now(timezone.utc) >= deadline:
        raise TimeoutError('48-hour deadline reached')
    if json.loads((folder / 'full-refinement-status.json').read_text())['status'] != 'complete':
        raise ValueError('Full refinement must finish before this comparison')
    plan = json.loads((folder / 'allocation-54344308416.json').read_text())
    prior = Path(plan['variants'])
    if not prior.is_absolute():
        prior = ROOT / prior
    output = folder / 'native-iq4-variants'
    output.mkdir(parents=True, exist_ok=True)
    source_db = sqlite3.connect(f'{(prior / "state.sqlite").as_uri()}?mode=ro', uri=True)
    connection = sqlite3.connect(output / 'state.sqlite')
    connection.execute('CREATE TABLE IF NOT EXISTS chunks(name TEXT,qtype TEXT,part INTEGER,offset INTEGER,size INTEGER,sha256 TEXT,metrics TEXT,PRIMARY KEY(name,qtype,part))')
    connection.execute('CREATE TABLE IF NOT EXISTS metadata(name TEXT PRIMARY KEY,value TEXT)')
    training = folder / 'capture-train-extended/imatrix.gguf'
    metadata = {
        'source_revision': REPO_REVISION,
        'method': 'Ordinary native calibrated IQ4_NL; preserve refined balanced Q2_0 and conservative weak IQ4 experts. Only IQ4 payload choice differs.',
        'builder_sha256': sha(Path(__file__)),
        'refinement_code_sha256': sha(ROOT / 'tools/refine_quantization.py'),
        'quantizer_dll_sha256': sha(ROOT / 'third_party/llama.cpp/build-ninja-release/bin/ggml-base.dll'),
        'imatrix_sha256': sha(training),
        'prior_state_sha256': sha(prior / 'state.sqlite'),
        'balanced_allocation_sha256': sha(folder / 'allocation-54344308416.json'),
        'deadline_utc': deadline.isoformat(),
    }
    for key, value in metadata.items():
        old = connection.execute('SELECT value FROM metadata WHERE name=?', (key,)).fetchone()
        if old and old[0] != value:
            raise ValueError(f'Resume provenance mismatch: {key}')
        connection.execute('INSERT OR REPLACE INTO metadata VALUES (?,?)', (key, value))
    connection.commit()
    importance_values, _ = read_imatrix(training)
    names = {name for name, fmt in plan['replacement_formats'].items() if fmt == 'IQ4_NL'}
    source_files = checked_source(ROOT / 'data/source', names)
    torch.set_num_threads(8)
    started = time.monotonic()
    completed = 0
    for name, fmt in plan['replacement_formats'].items():
        layer = int(name.split('.')[1])
        size = 2560 * 640 // (64 if fmt == 'Q2_0' else 32) * 18
        path = output / f'{name.replace(".", "_")}.{fmt}.bin'
        with path.open('r+b' if path.exists() else 'w+b') as target:
            target.truncate(512 * size)
            raw_source = safe_open(ROOT / 'data/source' / source_files[name], framework='pt', device='cpu') if fmt == 'IQ4_NL' else None
            with (raw_source if raw_source is not None else (prior / f'{name.replace(".", "_")}.{fmt}.bin').open('rb')) as handle:
                sliced = handle.get_slice(source_name(name)) if fmt == 'IQ4_NL' else None
                if sliced is not None and sliced.get_shape() != [512, 2560, 640]:
                    raise ValueError('Unexpected BF16 expert shape')
                for expert in range(512):
                    if datetime.now(timezone.utc) >= deadline:
                        raise TimeoutError('48-hour deadline reached; native comparison is resumable')
                    saved = connection.execute('SELECT sha256 FROM chunks WHERE name=? AND qtype=? AND part=?', (name, fmt, expert)).fetchone()
                    target.seek(expert * size)
                    if saved and hashlib.sha256(target.read(size)).hexdigest() == saved[0]:
                        continue
                    previous = source_db.execute('SELECT offset,size,sha256,metrics FROM chunks WHERE name=? AND qtype=? AND part=?', (name, fmt, expert)).fetchone()
                    if not previous or previous[:2] != (expert * size, size):
                        raise ValueError('Incomplete prior pool')
                    metrics = json.loads(previous[3])
                    if fmt == 'Q2_0' or metrics['experts_with_under_8_samples']:
                        with (prior / f'{name.replace(".", "_")}.{fmt}.bin').open('rb') as file:
                            file.seek(previous[0])
                            payload = file.read(size)
                        if hashlib.sha256(payload).hexdigest() != previous[2]:
                            raise ValueError('Prior payload checksum changed')
                        metrics['comparison_method'] = 'retained balanced Q2' if fmt == 'Q2_0' else 'retained conservative IQ4'
                    else:
                        weight = sliced[expert].float().contiguous()
                        if not torch.isfinite(weight).all():
                            raise ValueError('Nonfinite BF16 weights')
                        values, counts = importance_values[name]
                        factor = np.nan_to_num(values[expert] / max(float(counts[expert]), 1), nan=1., posinf=100., neginf=1.)
                        if counts[expert] == 0:
                            factor = np.ones(640, dtype=np.float32)
                        importance = torch.from_numpy(np.tile(np.clip(factor, 1e-8, 100.), (2560, 1)))
                        payload = native_quant(weight, fmt, importance).tobytes()
                        # The same deterministic native quantizer and calibration
                        # supplied this score in the completed refinement pool.
                        native_score = metrics['method_scores']['native_calibrated']
                        metrics.update(native_score)
                        metrics['selected_method'] = 'native_calibrated'
                        metrics['comparison_method'] = 'ordinary native calibrated IQ4'
                        metrics['allocation_error'] = native_score['output_sse'] / metrics['validation_vectors']
                    if len(payload) != size:
                        raise ValueError('Incorrect packed expert size')
                    digest = hashlib.sha256(payload).hexdigest()
                    target.seek(expert * size)
                    target.write(payload)
                    target.flush()
                    connection.execute('INSERT OR REPLACE INTO chunks VALUES (?,?,?,?,?,?,?)',
                                       (name, fmt, expert, expert * size, size, digest, json.dumps(metrics, sort_keys=True)))
                    connection.commit()
                    completed += 1
                    if expert % 64 == 0 or expert == 511:
                        progress = {'stage': 'ordinary native IQ4 comparison', 'layer': layer, 'expert': expert,
                                    'completed_this_run': completed, 'elapsed_seconds': time.monotonic() - started,
                                    'utc': datetime.now(timezone.utc).isoformat()}
                        (output / 'progress.json').write_text(json.dumps(progress, indent=2) + '\n')
                        print(json.dumps(progress), flush=True)
    if connection.execute('SELECT count(*) FROM chunks').fetchone()[0] != 24576:
        raise ValueError('Incomplete native comparison')
    connection.close()
    source_db.close()
    plan['variants'] = str(output)
    plan['objective'] = 'Ordinary native calibrated IQ4 comparison at unchanged bytes and unchanged refined Q2 weights; task tests decide quality.'
    plan.pop('layers', None)
    plan['precision_layout_source_sha256'] = metadata['balanced_allocation_sha256']
    plan['metric_note'] = 'Precision layout retained from refined balanced allocation; operator scores for these actual payloads are in native-iq4-variants/state.sqlite. Prior selected-payload layer scores are omitted.'
    (folder / 'allocation-native-iq4.json').write_text(json.dumps(plan, indent=2) + '\n')
    (output / 'status.json').write_text(json.dumps({'status': 'complete', 'chunks': 24576,
                                                'utc': datetime.now(timezone.utc).isoformat()}, indent=2) + '\n')
    print('Ordinary native comparison pool complete; no model assembled', flush=True)


if __name__ == '__main__':
    main()
