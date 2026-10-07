"""Compare already-published ISTA GSQ gate/up payloads to the mixed pilot reference.

Read-only comparison: no quantization, model assembly, or confirmation answers.
Run after the CUDA pilot releases the shared model lock.
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import pilot_gateup_block as pilot
from tools.assemble_gateup_trial import exclusive_model_lock
from tools.read_gateup_capture import verify_capture_linkage, sha256

SOURCE = ROOT / 'data/base/Q2_0/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf'
EXPECTED_SHA = '69820c02ec7d0b45ef2ebb19d6620299db749fe2aded7f39f93c6b88b199b720'
REVISION = 'ed59f92082b1e93c0e96d60a8b11aab089b52f09'
OUTPUT = pilot.FOLDER / 'published-q2-gateup'


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    if (OUTPUT / 'summary.json').exists():
        raise FileExistsError('Completed published-payload comparison exists')
    budget = json.loads((pilot.FOLDER / 'budget.json').read_text(encoding='utf-8-sig'))
    deadline = datetime.fromisoformat(budget['pilot_deadline_utc'])
    def check_deadline():
        if datetime.now(timezone.utc) >= deadline:
            raise TimeoutError('Bounded pilot deadline reached')
    check_deadline()
    metadata = (ROOT / 'data/base/.cache/huggingface/download/Q2_0' / (SOURCE.name + '.metadata')).read_text().splitlines()
    if metadata[:2] != [REVISION, EXPECTED_SHA]:
        raise ValueError('Cached source provenance differs from the pinned ISTA revision')
    before = SOURCE.stat()
    checksum = hashlib.sha256()
    pilot.write_json(OUTPUT / 'status.json', {'status': 'running', 'stage': 'Verifying complete pinned ISTA source', 'pid': __import__('os').getpid()})
    with SOURCE.open('rb') as stream:
        while block := stream.read(8 * 1024**2):
            check_deadline()
            checksum.update(block)
    if checksum.hexdigest() != EXPECTED_SHA or SOURCE.stat().st_mtime_ns != before.st_mtime_ns:
        raise ValueError('Complete ISTA source checksum failed')
    pilot.write_json(OUTPUT / 'source-verified.json', {'path': str(SOURCE), 'sha256': EXPECTED_SHA, 'revision': REVISION,
                     'bytes': before.st_size, 'mtime_ns': before.st_mtime_ns, 'verified_utc': datetime.now(timezone.utc).isoformat()})
    prior = json.loads((pilot.FOLDER / 'pilot/summary.json').read_text(encoding='utf-8'))
    if prior['confirmation_used']:
        raise ValueError('Unexpected confirmation use')
    identity = prior['control']
    control = Path(identity['path'])
    if control.stat().st_size != identity['bytes'] or control.stat().st_mtime_ns != identity['mtime_ns']:
        raise ValueError('Control changed')
    layers = [row['layer'] for row in prior['layers']]
    train, train_linkage = verify_capture_linkage(pilot.FOLDER / 'capture-train', 'train', pilot.FOLDER / 'calibration/manifest.json', identity, layers)
    valid, valid_linkage = verify_capture_linkage(pilot.FOLDER / 'capture-validation', 'validation', pilot.FOLDER / 'calibration/manifest.json', identity, layers)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required')
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    source_reader = pilot.gguf.GGUFReader(SOURCE)
    control_reader = pilot.gguf.GGUFReader(control)
    source_tensors = {t.name: t for t in source_reader.tensors}
    control_tensors = {t.name: t for t in control_reader.tensors}
    results = []
    started = time.monotonic()
    for row in prior['layers']:
        layer = row['layer']
        ref_path = Path(row['reference_arrays']['path'])
        if sha256(ref_path) != row['reference_arrays']['sha256']:
            raise ValueError('Reference arrays changed')
        with np.load(ref_path, allow_pickle=False) as reference:
            targets = [reference['teacher_train'], reference['teacher_validation']]
            controls = [reference['control_train'], reference['control_validation']]
            captures = (train[layer], valid[layer])
            outputs = [np.zeros_like(capture.y) for capture in captures]
            payload_hashes = {half: hashlib.sha256() for half in ('gate', 'up')}
            for expert in range(512):
                check_deadline()
                halves = []
                for half in ('gate', 'up'):
                    tensor = source_tensors[f'blk.{layer}.ffn_{half}_exps.weight']
                    if tensor.tensor_type.name != 'Q2_0':
                        raise ValueError('Published tensor is not the expected Q2_0 format')
                    decoded, packed = pilot.tensor_expert(tensor, expert, (2560, 640, 512))
                    payload_hashes[half].update(packed)
                    halves.append(decoded)
                down, _ = pilot.tensor_expert(control_tensors[f'blk.{layer}.ffn_down_exps.weight'], expert, (640, 2560, 512))
                for index, capture in enumerate(captures):
                    rows, probability = pilot.routes(capture, expert)
                    if len(rows):
                        outputs[index][rows] += pilot.numpy_contribution(capture.x[rows], *halves, down, probability, 'cuda')
                if expert % 64 == 0:
                    pilot.write_json(OUTPUT / 'status.json', {'status': 'running', 'layer': layer, 'expert': expert})
            scores = [pilot.metrics(output, target) for output, target in zip(outputs, targets)]
            control_scores = [pilot.metrics(output, target) for output, target in zip(controls, targets)]
            result = {'layer': layer, 'format': 'Q2_0', 'mse_ratio': [s['output_sse'] / c['output_sse'] for s, c in zip(scores, control_scores)],
                      'scores': scores, 'control_scores': control_scores, 'sha256': {half: value.hexdigest() for half, value in payload_hashes.items()}}
            results.append(result)
            pilot.write_json(OUTPUT / f'layer{layer}.json', result)
            print(json.dumps({'layer': layer, 'published_GSQ_Q2_mse_ratio': result['mse_ratio']}), flush=True)
    report = {'status': 'complete', 'method': 'Retained published ISTA GSQ-RCO Q2_0 gate/up; no fresh optimization',
              'source': {'path': str(SOURCE), 'sha256': EXPECTED_SHA, 'revision': REVISION, 'bytes': before.st_size, 'mtime_ns': before.st_mtime_ns},
              'control': identity, 'results': results, 'models_assembled': 0, 'confirmation_used': False,
              'train_linkage': train_linkage, 'validation_linkage': valid_linkage, 'seconds': time.monotonic() - started,
              'reference': prior['reference'], 'completed_utc': datetime.now(timezone.utc).isoformat(), 'script_sha256': sha256(Path(__file__))}
    pilot.write_json(OUTPUT / 'summary.json', report)
    pilot.write_json(OUTPUT / 'status.json', {'status': 'complete'})


if __name__ == '__main__':
    with exclusive_model_lock():
        main()
