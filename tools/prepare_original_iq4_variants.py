"""Verify original two-expert chunks and expose one-expert adapter records.

Payloads are hard links to the immutable original variant pool. Only the
index changes; no weights are re-quantized or original records modified.
"""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sqlite3

ROOT = Path(__file__).resolve().parents[1]
NAMES = ['blk.5.ffn_down_exps.weight', 'blk.13.ffn_down_exps.weight']
EXPERT_BYTES = 2560 * 640 // 32 * 18


def main():
    source = ROOT / 'data/variants'
    target = ROOT / 'data/optimization-48h/original-iq4-promotion-variants'
    deadline = datetime.fromisoformat(json.loads((ROOT / 'data/optimization-48h/budget.json').read_text())['deadline_utc'].replace('Z', '+00:00'))
    original_state = source / 'state.sqlite'
    old = sqlite3.connect(f'file:{original_state.as_posix()}?mode=ro', uri=True)
    metadata = dict(old.execute('SELECT name,value FROM metadata'))
    if metadata['source_revision'] != 'deb02632504bb214702bc28b0381a93d3112f500':
        raise ValueError('Original source revision differs')
    metadata.update(adapter='Original two-expert chunks split into one-expert checksum records; unchanged bytes',
                    original_state_sha256=hashlib.sha256(original_state.read_bytes()).hexdigest(),
                    adapter_code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    target.mkdir(parents=True, exist_ok=True)
    current = sqlite3.connect(target / 'state.sqlite')
    current.execute('CREATE TABLE IF NOT EXISTS chunks(name TEXT,qtype TEXT,part INTEGER,offset INTEGER,size INTEGER,sha256 TEXT,metrics TEXT,PRIMARY KEY(name,qtype,part))')
    current.execute('CREATE TABLE IF NOT EXISTS metadata(name TEXT PRIMARY KEY,value TEXT)')
    for key, value in metadata.items():
        prior = current.execute('SELECT value FROM metadata WHERE name=?', (key,)).fetchone()
        if prior and prior[0] != value:
            raise ValueError('Adapter provenance changed; inspect before resuming')
        current.execute('INSERT OR REPLACE INTO metadata VALUES (?,?)', (key, value))
    current.commit()
    for name in NAMES:
        original = source / (name.replace('.', '_') + '.IQ4_NL.bin')
        output = target / original.name
        if original.stat().st_size != 512 * EXPERT_BYTES:
            raise ValueError('Original IQ4 payload has unexpected size')
        rows = old.execute('SELECT part,offset,size,sha256,metrics FROM chunks WHERE name=? AND qtype=? ORDER BY part', (name, 'IQ4_NL')).fetchall()
        if len(rows) != 256 or any(part != i or offset != i * 2 * EXPERT_BYTES or size != 2 * EXPERT_BYTES for i, (part, offset, size, _, _) in enumerate(rows)):
            raise ValueError('Original two-expert index is incomplete')
        if output.exists():
            if not os.path.samefile(original, output):
                raise ValueError('Adapter payload is not the original hard link')
        else:
            os.link(original, output)
        with original.open('rb') as handle:
            for part, offset, size, digest, metrics in rows:
                if datetime.now(timezone.utc) >= deadline:
                    raise TimeoutError('48-hour deadline reached')
                payload = handle.read(size)
                if hashlib.sha256(payload).hexdigest() != digest:
                    raise ValueError('Original IQ4 checksum differs')
                for sub in (0, 1):
                    child = payload[sub * EXPERT_BYTES:(sub + 1) * EXPERT_BYTES]
                    expert = part * 2 + sub
                    record = json.loads(metrics)
                    record['adapter_original_part'] = part
                    record['adapter_note'] = 'Inherited two-expert metrics; not a fresh per-expert operator measurement'
                    current.execute('INSERT OR REPLACE INTO chunks VALUES (?,?,?,?,?,?,?)', (name, 'IQ4_NL', expert, expert * EXPERT_BYTES, EXPERT_BYTES, hashlib.sha256(child).hexdigest(), json.dumps(record)))
            if handle.read(1):
                raise ValueError('Original IQ4 trailing bytes')
        current.commit()
        print(f'Verified original IQ4 and indexed512 experts: {name}', flush=True)
    old.close()
    current.close()
    (target / 'status.json').write_text(json.dumps({'status': 'complete', 'layers': NAMES, 'source': str(source), 'scope': 'Original payloads, index adapter only; no new calibration or capability gain'}, indent=2) + '\n')


if __name__ == '__main__':
    main()
