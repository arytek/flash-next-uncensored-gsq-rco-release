"""Repair only the trial's Q8_0 SSM columns, with local backups and an audit."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sqlite3
from pathlib import Path

import numpy as np

from tools.assemble_calibrated import variant_path
from tools.quantize_edited import convert_dense_layout
from tools.verify_calibrated import equal_ranges
import gguf


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--allocation', type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.allocation.read_text(encoding='utf-8'))
    output = Path(plan['output_shards'][0])
    reader = gguf.GGUFReader(output)
    tensors = {t.name: t for t in reader.tensors}
    folder = Path('data/variants')
    backup = Path('data/variants-before-ssm-layout-fix')
    backup.mkdir(exist_ok=True)
    connection = sqlite3.connect(folder / 'state.sqlite')
    hp = json.loads(Path('data/source/config.json').read_text())['text_config']
    keys, values, width = (hp[k] for k in ('linear_num_key_heads', 'linear_num_value_heads', 'linear_value_head_dim'))
    if width % 32 or values % keys:
        raise ValueError('Q8 block alignment incompatible with head permutation')
    audit = []
    for name, qtype in plan['tensor_types'].items():
        if not name.endswith('.ssm_out.weight'):
            continue
        if qtype != 'Q8_0':
            raise ValueError('Repair requires Q8_0 SSM output tensors')
        source = variant_path(folder, name, qtype)
        original = backup / source.name
        if not original.exists():
            shutil.copy2(source, original)
        rows, columns = map(int, reversed(tensors[name].shape))
        blob = np.fromfile(original, dtype=np.uint8).reshape(rows, keys, values // keys, width // 32, 34)
        fixed = blob.transpose(0, 2, 1, 3, 4).copy().reshape(-1)
        # Independent numerical layout check, including exact Q8 block ordering.
        import torch
        decoded_old = gguf.quants.dequantize(blob.reshape(rows, -1), gguf.GGMLQuantizationType.Q8_0)
        decoded_new = gguf.quants.dequantize(fixed.reshape(rows, -1), gguf.GGMLQuantizationType.Q8_0)
        expected = convert_dense_layout(torch.from_numpy(decoded_old), name, hp).numpy()
        if not np.array_equal(decoded_new, expected):
            raise ValueError(f'Numerical layout mismatch: {name}')
        fixed.tofile(source)
        with source.open('rb') as payload:
            for part, offset, size in connection.execute(
                'SELECT part,offset,size FROM chunks WHERE name=? AND qtype=?', (name, qtype)).fetchall():
                payload.seek(offset)
                digest = hashlib.sha256(payload.read(size)).hexdigest()
                connection.execute('UPDATE chunks SET sha256=? WHERE name=? AND qtype=? AND part=?',
                    (digest, name, qtype, part))
        connection.commit()
        with output.open('r+b') as destination:
            destination.seek(tensors[name].data_offset)
            destination.write(fixed.tobytes())
            destination.flush()
        if not equal_ranges(source, 0, output, tensors[name].data_offset, fixed.nbytes):
            raise ValueError(f'Output repair verification failed: {name}')
        audit.append({'name': name, 'bytes': fixed.nbytes,
            'old_sha256': hashlib.sha256(original.read_bytes()).hexdigest(),
            'new_sha256': hashlib.sha256(fixed).hexdigest()})
        print(f'Repaired and verified {name}', flush=True)
    connection.execute('INSERT OR REPLACE INTO metadata VALUES (?,?)',
        ('ssm_layout', 'llama.cpp grouped-to-tiled V heads v1'))
    connection.commit()
    connection.close()
    Path('logs/ssm-layout-repair.json').write_text(json.dumps(audit, indent=2) + '\n')
    print(f'Repaired {len(audit)} tensors; all other output bytes untouched', flush=True)


if __name__ == '__main__':
    main()
