"""Verify retained reconstruction pools and provenance after final screening."""
import hashlib
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FOLDER = ROOT / 'data/optimization-48h'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def main():
    source = ROOT / 'data/source'
    manifest = read(source / 'text-source-manifest.json')
    assert manifest['revision'] == 'deb02632504bb214702bc28b0381a93d3112f500' and manifest['complete']
    assert len(manifest['files']) == 51
    for filename, record in manifest['files'].items():
        assert (source / filename).stat().st_size == record['bytes']
    verified = {}
    for pool_name, builder, expected_count in [
        ('dense-f16-variants', 'prepare_dense_precision.py', 960),
        ('q8-promotion-variants', 'prepare_q8_expert_promotion.py', 1024),
    ]:
        pool = FOLDER / pool_name
        connection = sqlite3.connect(f'file:{(pool / "state.sqlite").as_posix()}?mode=ro', uri=True)
        metadata = dict(connection.execute('SELECT name,value FROM metadata'))
        assert metadata['builder_sha256'] == digest(ROOT / 'tools' / builder)
        assert metadata['source_revision'] == manifest['revision']
        assert metadata['source_manifest_sha256'] == digest(source / 'text-source-manifest.json')
        assert metadata['source_index_sha256'] == digest(source / 'model.safetensors.index.json')
        entries = connection.execute('SELECT name,qtype,part,offset,size,sha256 FROM chunks ORDER BY name,qtype,part').fetchall()
        assert len(entries) == expected_count
        grouped = {}
        for name, qtype, part, offset, size, sha in entries:
            grouped.setdefault((name, qtype), []).append((part, offset, size, sha))
        checked_bytes = 0
        for (name, qtype), parts in grouped.items():
            path = pool / (name.replace('.', '_') + '.' + qtype + '.bin')
            with path.open('rb') as stream:
                position = 0
                for expected_part, (part, offset, size, sha) in enumerate(parts):
                    assert part == expected_part and offset == position
                    raw = stream.read(size)
                    assert len(raw) == size and hashlib.sha256(raw).hexdigest() == sha
                    position += size
                assert not stream.read(1)
            assert path.stat().st_size == position
            checked_bytes += position
        connection.close()
        verified[pool_name] = {'chunks_verified': len(entries), 'tensors': len(grouped),
                               'payload_bytes_verified': checked_bytes, 'metadata': metadata,
                               'state_sha256': digest(pool / 'state.sqlite')}
    budget = read(FOLDER / 'budget.json')
    assert budget['started_utc'] == '2026-09-30T18:08:13Z'
    assert budget['deadline_utc'] == '2026-10-02T18:08:13Z'
    for filename in ['confirmation-status.json', 'q8-confirmation-status.json']:
        assert read(FOLDER / filename)['status'] == 'skipped-no-improvement'
    result = {'utc': datetime.now(timezone.utc).isoformat(), 'status': 'complete',
              'candidate_selected': False, 'pools': verified,
              'source_revision': manifest['revision'], 'retained_source_files': len(manifest['files']),
              'source_size_checks_passed': True,
              'source_full_hash_note': 'Source shard lengths and saved manifest verified here; prior build source checks retained, not a fresh whole-shard rehash.',
              'immutable_deadline_verified': True, 'conditional_confirmations_remain_skipped': True,
              'model_loads_or_generated_code_execution': False}
    (FOLDER / 'final-retained-source-audit.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'pools'}))


if __name__ == '__main__':
    main()
