"""Keep control IQ4/Q2 tensors where possible; replace only added Q2 layers."""
import argparse
import hashlib
import json
from pathlib import Path
from tools.assemble_refined_trial import predicted_first_shard_bytes, gguf


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    source = json.loads(args.source.read_text())
    if not source['feasible']:
        raise ValueError('Source allocation is infeasible')
    reader = gguf.GGUFReader(source['control'])
    tensors = {t.name: t for t in reader.tensors}
    replacements, retained = {}, {}
    for name, fmt in source['replacement_formats'].items():
        prior = tensors[name].tensor_type.name
        if fmt == prior:
            retained[name] = prior
        elif fmt == 'Q2_0' and prior == 'IQ4_NL':
            replacements[name] = fmt
        else:
            raise ValueError(f'Unexpected precision change: {name} {prior} -> {fmt}')
    if len(retained) + len(replacements) != 48 or not replacements:
        raise ValueError('Expected a complete, smaller precision allocation')
    predicted, header = predicted_first_shard_bytes(reader, replacements, args.output.with_suffix('.header.bin'))
    if predicted != source['predicted_transformer_bytes']:
        raise ValueError('Control-preserving allocation changed byte prediction')
    result = {k: source[k] for k in ('feasible', 'control', 'variants', 'transformer_budget_bytes', 'predicted_transformer_bytes', 'predicted_total_bytes', 'formats', 'mandatory_iq4_layers')}
    result.update(
        replacement_formats=replacements, retained_control_tensors=retained,
        header_bytes=header, source_allocation_sha256=hashlib.sha256(args.source.read_bytes()).hexdigest(),
        retained_control_transformer_sha256='84512ed5aaa14930c56345eaaa88adccddce5a80a700b102c0b9fbfdc955f507',
        objective='Smaller control-based candidate: preserve all existing IQ4 and original Q2 payloads; only added Q2 layers use refined variants. Format allocation uses routed operator error, not whole-model task loss.',
        metric_note='Prior layer scores omitted because retained IQ4 payloads differ from the refined pool. No capability or speed gain is predicted.',
    )
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'output': str(args.output), 'changed_layers': sorted(replacements), 'retained_expert_layers': len(retained), 'total_bytes': result['predicted_total_bytes']}))


if __name__ == '__main__':
    main()
