"""Promote only the control's two Q2 expert-down layers to calibrated IQ4_NL.

Retain the other 46 expert layers and all other tensor payloads. This trades a
small size increase for a measured test of quality and generation speed.
"""
import argparse
import hashlib
import json
from pathlib import Path

from tools.assemble_refined_trial import CONTROL, gguf, predicted_first_shard_bytes

ROOT = Path(__file__).resolve().parents[1]


def main():
    folder = ROOT / "data/optimization-48h"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variants', type=Path, default=folder / 'variants-extended')
    parser.add_argument('--output', type=Path, default=folder / 'allocation-control-iq4-down.json')
    parser.add_argument('--original-calibration', action='store_true')
    args = parser.parse_args()
    prior_path = folder / "allocation-54344308416.json"
    prior = json.loads(prior_path.read_text())
    if not prior["feasible"] or Path(prior["control"]).resolve() != CONTROL.resolve():
        raise ValueError("Unexpected balanced source plan")
    reader = gguf.GGUFReader(CONTROL)
    experts = {t.name: t for t in reader.tensors if t.name.endswith(".ffn_down_exps.weight")}
    q2_names = {n for n, t in experts.items() if t.tensor_type == gguf.GGMLQuantizationType.Q2_0}
    if len(experts) != 48 or q2_names != {"blk.5.ffn_down_exps.weight", "blk.13.ffn_down_exps.weight"}:
        raise ValueError("Expected 46 IQ4 and two Q2 control expert-down layers")
    if any(t.tensor_type != gguf.GGMLQuantizationType.IQ4_NL for n, t in experts.items() if n not in q2_names):
        raise ValueError("Unexpected retained control expert format")
    replacements = {n: "IQ4_NL" for n in sorted(q2_names)}
    output = args.output
    description = ('Local trial: retained ISTA GSQ-RCO weights and corrected abliterated control. Only expert-down layers 5 and 13 promoted from Q2_0 to IQ4_NL using the original BF16 calibration and Gumbel/scale adapter. Other payloads unchanged. Targeted adaptation, not full upstream GSQ/RCO.' if args.original_calibration else None)
    predicted, header = predicted_first_shard_bytes(reader, replacements, output.with_suffix(".header.bin"), description)
    expected = CONTROL.stat().st_size + 2 * (512 * 2560 * 640 // 64 * 18) + header - min(t.data_offset for t in reader.tensors)
    if predicted != expected:
        raise ValueError("Promotion byte count differs from serialized GGUF prediction")
    lookup = Path(str(CONTROL).replace("-00001-of-00002", "-00002-of-00002"))
    plan = {
        "feasible": True,
        "control": str(CONTROL),
        "variants": str(args.variants.resolve()),
        "replacement_formats": replacements,
        "retained_control_tensors": {n: t.tensor_type.name for n, t in experts.items() if n not in q2_names},
        "retained_control_transformer_sha256": "84512ed5aaa14930c56345eaaa88adccddce5a80a700b102c0b9fbfdc955f507",
        "transformer_budget_bytes": predicted,
        "predicted_transformer_bytes": predicted,
        "predicted_total_bytes": predicted + lookup.stat().st_size,
        "header_bytes": header,
        "formats": {"IQ4_NL": 48, "Q2_0": 0},
        "source_allocation_sha256": hashlib.sha256(prior_path.read_bytes()).hexdigest(),
        "objective": "Retain current control payloads except promote Q2 expert-down layers 5 and 13 to independently calibrated IQ4_NL. Test precision and kernel tradeoff without replacing the other 46 expert layers.",
        "metric_note": "No whole-model quality or speed gain is predicted; capability and fixed runtime tests decide the result. All-IQ4 applies only to expert-down tensors, not the entire model.",
    }
    if description:
        plan['metadata_description'] = description
        plan['objective'] = 'Isolate the two-layer precision change using original-control IQ4 payloads rather than the extended routed refinement. Retain the other 46 control expert layers and all other tensors.'
        plan['metric_note'] = 'Original calibration, not the new routed operator calibration. No whole-model gain predicted; fixed capability and runtime comparisons decide the result.'
    output.write_text(json.dumps(plan, indent=2) + "\n")
    print(json.dumps({"allocation": str(output), "changed_layers": sorted(replacements), "transformer_bytes": predicted, "total_bytes": plan["predicted_total_bytes"]}))


if __name__ == "__main__":
    main()
