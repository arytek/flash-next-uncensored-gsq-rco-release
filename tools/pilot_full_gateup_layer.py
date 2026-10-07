"""Check complete uniform-format gate/up tensors against the frozen mixed reference.

All512 experts are freshly packed. Only the preselected12 covered experts may
receive training-only scale updates; other experts keep the native candidate,
never higher-format control bytes. This does not assemble or accept a model.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import pilot_gateup_block as pilot
from tools.read_gateup_capture import sha256, verify_capture_linkage, verify_calibration_manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prototype', type=Path, default=pilot.FOLDER / 'pilot')
    parser.add_argument('--output', type=Path, default=pilot.FOLDER / 'full-layer-pilot')
    parser.add_argument('--formats', nargs='+', choices=pilot.FORMATS,
                        default=['IQ2_S', 'IQ2_XS', 'IQ3_XXS'])
    args = parser.parse_args()
    if not args.output.resolve().is_relative_to(pilot.FOLDER.resolve()):
        raise ValueError('Full layer outputs must stay in the private phase directory')
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / 'summary.json').exists():
        raise FileExistsError('Completed full layer pilot exists; do not rerun it')
    if any(args.output.glob('*.bin')):
        raise FileExistsError('Partial packed files exist; preserve and inspect before a retry')
    budget = json.loads((pilot.FOLDER / 'budget.json').read_text(encoding='utf-8-sig'))
    if budget['status'].startswith('complete'):
        raise RuntimeError('Final optimisation phase is closed; no pilot restart')
    deadline = datetime.fromisoformat(budget['pilot_deadline_utc'])
    def check_deadline():
        if datetime.now(timezone.utc) >= deadline:
            raise TimeoutError('Final pilot deadline reached')
    check_deadline()
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required; no silent runtime fallback')
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    prior = json.loads((args.prototype / 'summary.json').read_text(encoding='utf-8'))
    if prior['status'] != 'complete' or prior['confirmation_used']:
        raise ValueError('Full layer check requires a complete unconfirmed prototype')
    identity = prior['control']
    control = Path(identity['path'])
    if control.stat().st_size != identity['bytes'] or control.stat().st_mtime_ns != identity['mtime_ns']:
        raise ValueError('Preserved control identity changed')
    calibration = verify_calibration_manifest(pilot.FOLDER / 'calibration/manifest.json')
    layers = [row['layer'] for row in prior['layers']]
    train_captures, train_linkage = verify_capture_linkage(pilot.FOLDER / 'capture-train', 'train',
                        pilot.FOLDER / 'calibration/manifest.json', identity, layers)
    valid_captures, valid_linkage = verify_capture_linkage(pilot.FOLDER / 'capture-validation', 'validation',
                        pilot.FOLDER / 'calibration/manifest.json', identity, layers)
    for key in ('calibration_cpu_moe', 'calibration_load_mode', 'engine_sha256', 'collector_sha256'):
        if train_linkage[key] != valid_linkage[key]:
            raise ValueError('Capture runtime declarations disagree')
    importance = pilot.load_imatrix(Path(train_linkage['imatrix_path']), layers, train_linkage)
    reader = pilot.gguf.GGUFReader(control)
    tensors = {tensor.name: tensor for tensor in reader.tensors}
    native = pilot.NativeQuantizer()
    started = time.monotonic()
    manifest = {'method': 'Complete native calibrated IQ gate/up tensors with optional12-expert training-only global-scale updates',
        'full_GSQ_RCO': False, 'models_assembled': 0, 'confirmation_used': False,
        'reference': prior['reference'], 'control': identity, 'calibration': calibration,
        'prototype_summary_sha256': sha256(args.prototype / 'summary.json'),
        'builder_sha256': sha256(Path(__file__)), 'pilot_builder_sha256': sha256(Path(pilot.__file__)),
        'train_linkage': train_linkage, 'validation_linkage': valid_linkage,
        'pid': os.getpid(), 'started_utc': datetime.now(timezone.utc).isoformat()}
    pilot.write_json(args.output / 'manifest.json', manifest)
    results = []
    for row in prior['layers']:
        layer = row['layer']
        check_deadline()
        reference_path = Path(row['reference_arrays']['path'])
        if sha256(reference_path) != row['reference_arrays']['sha256']:
            raise ValueError('Mixed-reference arrays changed')
        references = np.load(reference_path, allow_pickle=False)
        captures = (train_captures[layer], valid_captures[layer])
        targets = [references['teacher_train'], references['teacher_validation']]
        controls = [references['control_train'], references['control_validation']]
        if any(target.shape != capture.y.shape or not np.isfinite(target).all()
               for target, capture in zip(targets, captures)):
            raise ValueError('Reference shapes or finite values failed')
        control_scores = [pilot.metrics(prediction, target) for prediction, target in zip(controls, targets)]
        source_record = prior['sources'][str(layer)]
        source_path = Path(source_record['path'])
        if sha256(source_path) != source_record['sha256']:
            raise ValueError('Pinned source changed since prototype')
        counts_train, mass_train = captures[0].coverage()
        counts_valid, _ = captures[1].coverage()
        for fmt in args.formats:
            check_deadline()
            fmt_started = time.monotonic()
            kind = getattr(pilot.gguf.GGMLQuantizationType, fmt)
            block, block_bytes = pilot.gguf.GGML_QUANT_SIZES[kind]
            expert_bytes = 640 * 2560 // block * block_bytes
            paths = {half: args.output / f'layer{layer}-{fmt}-{half}.bin' for half in ('gate', 'up')}
            outputs = [np.zeros_like(capture.y) for capture in captures]
            receipts = []
            with safe_open(source_path, framework='pt', device='cpu') as source:
                fused = source.get_slice(source_record['tensor'])
                with paths['gate'].open('wb') as gate_file, paths['up'].open('wb') as up_file:
                    files = (gate_file, up_file)
                    for expert in range(512):
                        check_deadline()
                        bf = fused[expert].float().numpy()
                        packed = []
                        for half, weights, stream in zip(('gate', 'up'), (bf[:640], bf[640:]), files):
                            values, counts = importance[(layer, half)]
                            if counts[expert] > 0:
                                factors = values[expert] / float(counts[expert])
                            else:
                                factors = values.sum(0) / max(float(counts.sum()), 1.)
                            factors = np.clip(factors, 1e-8, 1e8).astype(np.float32)
                            blob = native.quantize(weights, fmt, factors)
                            if blob.nbytes != expert_bytes:
                                raise ValueError('Native full expert byte count mismatch')
                            stream.write(blob.tobytes())
                            packed.append(blob)
                        gate, up = [pilot.decode(blob, fmt) for blob in packed]
                        down, _ = pilot.tensor_expert(tensors[f'blk.{layer}.ffn_down_exps.weight'], expert, (640, 2560, 512))
                        for index, capture in enumerate(captures):
                            rows, probability = pilot.routes(capture, expert)
                            if len(rows):
                                outputs[index][rows] += pilot.numpy_contribution(capture.x[rows], gate, up, down, probability, 'cuda')
                        receipts.append({'expert': expert, 'offset': expert * expert_bytes, 'bytes': expert_bytes,
                            'method': 'native_calibrated', 'train_occurrences': int(counts_train[expert]),
                            'validation_occurrences': int(counts_valid[expert]),
                            'train_route_mass': float(mass_train[expert]),
                            'sha256': {half: hashlib.sha256(blob).hexdigest() for half, blob in zip(('gate', 'up'), packed)}})
                        if expert % 32 == 0:
                            pilot.write_json(args.output / 'status.json', {'status': 'running', 'stage': 'all512native-expert packing',
                                'layer': layer, 'format': fmt, 'expert': expert, 'elapsed_seconds': time.monotonic() - started})
                            print(json.dumps({'layer': layer, 'format': fmt, 'expert': expert}), flush=True)
                native_scores = [pilot.metrics(output, target) for output, target in zip(outputs, targets)]
                updates = []
                for selected in row['selected_experts']:
                    check_deadline()
                    expert = selected['expert']
                    blobs = []
                    for half in ('gate', 'up'):
                        with paths[half].open('rb') as stream:
                            stream.seek(expert * expert_bytes)
                            blobs.append(np.frombuffer(stream.read(expert_bytes), dtype=np.uint8).copy())
                    gate, up = [pilot.decode(blob, fmt) for blob in blobs]
                    bf = fused[expert].float().numpy()
                    down, _ = pilot.tensor_expert(tensors[f'blk.{layer}.ffn_down_exps.weight'], expert, (640, 2560, 512))
                    routing = [pilot.routes(capture, expert) for capture in captures]
                    old = [pilot.numpy_contribution(capture.x[rows], gate, up, down, p, 'cuda')
                           for capture, (rows, p) in zip(captures, routing)]
                    bases = [output[rows] - contribution for output, (rows, _), contribution in zip(outputs, routing, old)]
                    rows, p = routing[0]
                    before_training = pilot.metrics(outputs[0][rows], targets[0][rows])
                    check_deadline()
                    refined, trace = pilot.refine_scales(*blobs, fmt, captures[0].x[rows], p, down, bases[0],
                                                targets[0][rows], 64, 32, 'cuda', 1827 + layer * 512 + expert,
                                                check_deadline=check_deadline)
                    new_gate, new_up = [pilot.decode(blob, fmt) for blob in refined]
                    changed = [pilot.numpy_contribution(capture.x[rows], new_gate, new_up, down, p, 'cuda')
                               for capture, (rows, p) in zip(captures, routing)]
                    after_training = pilot.metrics(bases[0] + changed[0], targets[0][rows])
                    accepted = after_training['output_sse'] < before_training['output_sse']
                    if accepted:
                        for half, blob in zip(('gate', 'up'), refined):
                            with paths[half].open('r+b') as stream:
                                stream.seek(expert * expert_bytes)
                                stream.write(blob.tobytes())
                        for index, (rows, _) in enumerate(routing):
                            outputs[index][rows] = bases[index] + changed[index]
                        receipts[expert]['method'] = 'native_calibrated_scale_refined'
                        receipts[expert]['sha256'] = {half: hashlib.sha256(blob).hexdigest() for half, blob in zip(('gate', 'up'), refined)}
                    updates.append({'expert': expert, 'training_selected': accepted, 'trace': trace})
            final_scores = [pilot.metrics(output, target) for output, target in zip(outputs, targets)]
            packed_receipts = {}
            for half, path in paths.items():
                if path.stat().st_size != expert_bytes * 512:
                    raise ValueError('Full tensor does not contain exactly512complete experts')
                with path.open('rb') as stream:
                    for record in receipts:
                        data = stream.read(expert_bytes)
                        if hashlib.sha256(data).hexdigest() != record['sha256'][half]:
                            raise ValueError('Final expert payload checksum mismatch')
                packed_receipts[half] = {'path': str(path), 'sha256': sha256(path), 'bytes': path.stat().st_size,
                    'format': fmt, 'shape_ggml': [2560, 640, 512], 'experts': 512}
            result = {'layer': layer, 'format': fmt, 'control_error': control_scores, 'native_error': native_scores,
                'refined_error': final_scores, 'refined_to_control_mse_ratio': [new['output_sse'] / max(old['output_sse'], 1e-20)
                                     for new, old in zip(final_scores, control_scores)],
                'packing_and_scale_seconds': time.monotonic() - fmt_started, 'packed_tensors': packed_receipts,
                'experts': receipts, 'updates': updates, 'poorly_covered_train_experts': int(np.sum(counts_train < 32)),
                'deployment_scope': 'Uniform whole-format512-expert tensors; poorly covered experts use native calibrated fallback, no retained higher-format bytes',
                'whole_model_acceptance': False}
            pilot.write_json(args.output / f'layer{layer}-{fmt}.json', result)
            results.append(result)
            print(json.dumps({'layer': layer, 'format': fmt, 'mse_ratio': result['refined_to_control_mse_ratio']}), flush=True)
            del outputs
            gc.collect()
            torch.cuda.empty_cache()
        references.close()
    summary = {**manifest, 'status': 'complete', 'results': results, 'elapsed_seconds': time.monotonic() - started,
               'completed_utc': datetime.now(timezone.utc).isoformat(), 'whole_model_acceptance': False}
    pilot.write_json(args.output / 'summary.json', summary)
    pilot.write_json(args.output / 'status.json', {'status': 'complete', 'summary': str(args.output / 'summary.json')})


if __name__ == '__main__':
    main()
