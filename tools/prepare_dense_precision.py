"""Bounded local F16 variants for the 48 edited output projections.

Reconstruct each control Q8 chunk from the pinned BF16 source before changing
precision. This checks the SSM head permutation and the source mapping. Keep
all expert weights and the lookup shard unchanged. No speed/quality claim.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sqlite3
import struct
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'third_party/llama.cpp/gguf-py'))
import gguf
import numpy as np
from tools.assemble_calibrated import variant_path
from tools.assemble_refined_trial import CONTROL
from tools.download_ablated_text import REVISION, selected_shards


def digest(data):
    return hashlib.sha256(data).hexdigest()


def source_name(name):
    layer = int(name.split('.')[1])
    projection = 'self_attn.o_proj.weight' if layer % 4 == 3 else 'linear_attn.out_proj.weight'
    expected = f'blk.{layer}.' + ('attn_output.weight' if layer % 4 == 3 else 'ssm_out.weight')
    if name != expected:
        raise ValueError('Only edited attention/SSM output projections are allowed')
    return f'model.language_model.layers.{layer}.{projection}'


def read_header(path):
    with path.open('rb') as handle:
        raw = handle.read(8)
        size = struct.unpack('<Q', raw)[0]
        if not 0 < size <= 16 * 1024 * 1024:
            raise ValueError('Invalid safetensors header size')
        encoded = handle.read(size)
    return json.loads(encoded), 8 + size, digest(raw + encoded)


def convert_layout(matrix, name, config):
    if not name.endswith('.ssm_out.weight'):
        return matrix
    hp = config.get('text_config', config)
    keys, values, width = (int(hp[k]) for k in
                          ('linear_num_key_heads', 'linear_num_value_heads', 'linear_value_head_dim'))
    if values % keys or matrix.shape[1] != values * width:
        raise ValueError('Invalid SSM head layout')
    return np.ascontiguousarray(matrix.reshape(len(matrix), keys, values // keys, width)
                               .transpose(0, 2, 1, 3).reshape(matrix.shape))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=ROOT / 'data/source')
    parser.add_argument('--output', type=Path, default=ROOT / 'data/optimization-48h/dense-f16-variants')
    parser.add_argument('--layers', default='0,23,47', help='Pilot by default; all for the full pool')
    parser.add_argument('--rows', type=int, default=128)
    args = parser.parse_args()
    if args.rows != 128:
        parser.error('Frozen protocol uses 128-row bounded chunks')
    budget = json.loads((ROOT / 'data/optimization-48h/budget.json').read_text(encoding='utf-8-sig'))
    deadline = datetime.fromisoformat(budget['deadline_utc'].replace('Z', '+00:00'))
    def check_deadline():
        if datetime.now(timezone.utc) >= deadline:
            raise TimeoutError('48-hour optimisation deadline reached; pool is resumable')
    check_deadline()
    layers = list(range(48)) if args.layers == 'all' else list(map(int, args.layers.split(',')))
    if len(set(layers)) != len(layers) or any(i < 0 or i >= 48 for i in layers):
        raise ValueError('Invalid output projection layers')
    names = [f'blk.{i}.' + ('attn_output.weight' if i % 4 == 3 else 'ssm_out.weight') for i in layers]
    ablated_names, _ = selected_shards(args.source)
    if any(source_name(n) not in ablated_names for n in names):
        raise ValueError('Projection is outside the pinned ablation manifest')
    index_path = args.source / 'model.safetensors.index.json'
    config_path = args.source / 'config.json'
    manifest_path = args.source / 'text-source-manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    if manifest.get('revision') != REVISION or not manifest.get('complete'):
        raise ValueError('Pinned edited BF16 source manifest is incomplete')
    mapping = json.loads(index_path.read_text(encoding='utf-8'))['weight_map']
    config = json.loads(config_path.read_text(encoding='utf-8'))
    control = gguf.GGUFReader(CONTROL)
    tensors = {t.name: t for t in control.tensors}
    args.output.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(args.output / 'state.sqlite')
    connection.execute('CREATE TABLE IF NOT EXISTS chunks (name TEXT,qtype TEXT,part INTEGER,offset INTEGER,size INTEGER,sha256 TEXT,metrics TEXT,PRIMARY KEY(name,qtype,part))')
    connection.execute('CREATE TABLE IF NOT EXISTS metadata (name TEXT PRIMARY KEY,value TEXT)')
    metadata = {
        'method': 'F16 edited output projections, BF16 source with exact control Q8 reconstruction gate; no new expert quantization',
        'source_revision': REVISION,
        'source_manifest_sha256': digest(manifest_path.read_bytes()),
        'source_index_sha256': digest(index_path.read_bytes()),
        'source_config_sha256': digest(config_path.read_bytes()),
        'control_sha256': '84512ed5aaa14930c56345eaaa88adccddce5a80a700b102c0b9fbfdc955f507',
        'ssm_layout': 'grouped-to-tiled V heads v1, checked by byte equality to corrected control Q8',
        'rows_per_part': '128', 'builder_sha256': digest(Path(__file__).read_bytes()),
        'scope': 'Operator reconstruction checks only; independent whole-model quality and speed remain required',
    }
    for key, value in metadata.items():
        previous = connection.execute('SELECT value FROM metadata WHERE name=?', (key,)).fetchone()
        if previous and previous[0] != value:
            raise ValueError(f'Resume metadata changed: {key}')
        connection.execute('INSERT OR IGNORE INTO metadata VALUES (?,?)', (key, value))
    connection.commit()
    for name in names:
        check_deadline()
        tensor = tensors[name]
        if tensor.tensor_type != gguf.GGMLQuantizationType.Q8_0:
            raise ValueError('Expected control Q8 output projection')
        hf_name = source_name(name)
        filename = mapping[hf_name]
        if Path(filename).name != filename:
            raise ValueError('Unsafe shard filename')
        shard = args.source / filename
        header, payload_offset, header_sha = read_header(shard)
        entry = header[hf_name]
        shape = tuple(entry['shape'])
        begin, end = entry['data_offsets']
        if entry['dtype'] != 'BF16' or shape != tuple(reversed(tensor.shape)) or len(shape) != 2:
            raise ValueError(f'Source/control shape or dtype mismatch: {name}')
        if begin < 0 or end - begin != math.prod(shape) * 2 or payload_offset + end > shard.stat().st_size:
            raise ValueError('BF16 source byte bounds are invalid')
        if shard.stat().st_size != manifest['files'][filename]['bytes']:
            raise ValueError('Source shard length differs from inspected manifest')
        rows, columns = shape
        row_bytes = columns * 2
        q8_row_bytes = columns // 32 * 34
        output = variant_path(args.output, name, 'F16')
        size = rows * row_bytes
        with shard.open('rb') as source, output.open('r+b' if output.exists() else 'w+b') as destination:
            if output.stat().st_size not in (0, size):
                raise ValueError('Unexpected existing F16 payload size')
            destination.truncate(size)
            for part, first in enumerate(range(0, rows, args.rows)):
                check_deadline()
                count = min(args.rows, rows - first)
                source.seek(payload_offset + begin + first * row_bytes)
                raw = source.read(count * row_bytes)
                if len(raw) != count * row_bytes:
                    raise ValueError('Truncated BF16 source')
                matrix = (np.frombuffer(raw, dtype='<u2').astype(np.uint32) << 16).view(np.float32).reshape(count, columns)
                matrix = convert_layout(matrix, name, config)
                if not np.isfinite(matrix).all():
                    raise ValueError('Non-finite BF16 source values')
                q8 = gguf.quants.quantize(matrix, gguf.GGMLQuantizationType.Q8_0)
                retained = tensor.data.reshape(-1)[first * q8_row_bytes:(first + count) * q8_row_bytes]
                if not np.array_equal(q8.reshape(-1), retained):
                    raise ValueError(f'Source/layout fails exact control Q8 reconstruction: {name} part{part}')
                fp16 = matrix.astype('<f2')
                if not np.isfinite(fp16).all():
                    raise ValueError('F16 conversion overflow')
                encoded = fp16.tobytes(order='C')
                decoded = gguf.quants.dequantize(q8, gguf.GGMLQuantizationType.Q8_0)
                q8_sse = float(np.square(decoded - matrix, dtype=np.float64).sum())
                f16_sse = float(np.square(fp16.astype(np.float32) - matrix, dtype=np.float64).sum())
                if f16_sse > q8_sse or not np.isfinite(f16_sse):
                    raise ValueError('F16 reconstruction is worse than the retained Q8 chunk')
                offset = first * row_bytes
                metrics = {'bf16_source_sha256': digest(raw), 'source_header_sha256': header_sha,
                           'control_q8_sha256': digest(q8.tobytes()), 'source_power': float(np.square(matrix, dtype=np.float64).sum()),
                           'q8_sse': q8_sse, 'selected_sse': f16_sse, 'weighted_sse': f16_sse,
                           'q8_control_reconstructed_exactly': True, 'rows': count}
                record = (offset, len(encoded), digest(encoded), json.dumps(metrics, sort_keys=True))
                previous = connection.execute('SELECT offset,size,sha256,metrics FROM chunks WHERE name=? AND qtype=? AND part=?', (name, 'F16', part)).fetchone()
                if previous:
                    destination.seek(offset)
                    if tuple(previous) != record or digest(destination.read(len(encoded))) != record[2]:
                        raise ValueError('Resume source/metrics/payload differ')
                else:
                    destination.seek(offset)
                    destination.write(encoded)
                    destination.flush()
                    connection.execute('INSERT INTO chunks VALUES (?,?,?,?,?,?,?)', (name, 'F16', part, *record))
                    connection.commit()
        print(f'Completed {name}: all source/layout chunks match control Q8; F16 reconstruction no worse', flush=True)
    records = connection.execute('SELECT name,metrics FROM chunks WHERE qtype=?', ('F16',)).fetchall()
    covered = sorted({name for name, _ in records})
    summaries = [json.loads(metrics) for _, metrics in records]
    report = {'utc': datetime.now(timezone.utc).isoformat(), 'status': 'complete', 'tensors': covered,
              'completed_chunks': len(records), 'q8_sse': sum(m['q8_sse'] for m in summaries),
              'f16_sse': sum(m['selected_sse'] for m in summaries), 'metadata': metadata,
              'scope': 'Compression error only; not model quality, speed, or acceptance',
              'deadline_utc': budget['deadline_utc']}
    (args.output / 'status.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8', newline='\n')
    print(json.dumps({k: report[k] for k in ('status', 'completed_chunks', 'q8_sse', 'f16_sse', 'scope')}), flush=True)
    connection.close()


if __name__ == '__main__':
    main()
