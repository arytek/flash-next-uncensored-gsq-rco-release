import unittest
import hashlib
from pathlib import Path

import gguf
import numpy as np

from tools.assemble_hybrid import audit, edited_names, write_output
from tools.verify_release import verify
from tools.fetch_edited_ranges import verify_ranges


def make_gguf(path: Path, *, donor: bool):
    names = sorted(edited_names())
    writer = gguf.GGUFWriter(path, "qwen4exp", split_max_tensors=0 if donor else len(names))
    writer.add_key_value("qwen4exp.block_count", 48, gguf.GGUFValueType.UINT32)
    writer.add_key_value(
        "qwen4exp.attention.compress_ratios",
        [4 if layer % 4 == 3 else 0 for layer in range(48)],
        gguf.GGUFValueType.ARRAY,
        gguf.GGUFValueType.UINT32,
    )
    writer.add_key_value("general.name", "donor" if donor else "base", gguf.GGUFValueType.STRING)
    writer.add_key_value("general.alignment", 32, gguf.GGUFValueType.UINT32)
    for name in names:
        writer.add_tensor(name, np.full((4,), 2 if donor else 1, dtype=np.float32))
    if donor:
        writer.add_tensor("untouched.weight", np.full((4,), 9, dtype=np.float32))
    else:
        writer.add_tensor("untouched.weight", np.full((1, 18), 3, dtype=np.uint8),
                          raw_dtype=gguf.GGMLQuantizationType.Q2_0)
    if not donor:
        writer.add_tensor("per_layer_token_embd.weight", np.full((4,), 4, dtype=np.float32))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    return writer.format_shard_names(path)


class AssemblyTest(unittest.TestCase):
    def test_range_manifest_detects_changed_payload(self):
        folder = Path(__file__).resolve().parent / ".tmp-assembly"
        folder.mkdir(exist_ok=True)
        path = folder / "range-check.bin"
        path.write_bytes(b"header" + b"selected-tensor")
        payload = b"selected-tensor"
        tensors = {"sample.weight": (6, len(payload))}
        completed = {"sample.weight": {"offset": 6, "bytes": len(payload),
                                          "sha256": hashlib.sha256(payload).hexdigest()}}
        verify_ranges(path, tensors, completed)
        path.write_bytes(b"header" + b"changed-tensor!")
        with self.assertRaisesRegex(ValueError, "checksum mismatch"):
            verify_ranges(path, tensors, completed)

    def test_packed_tensor_selection_and_two_shard_layout(self):
        folder = Path(__file__).resolve().parent / ".tmp-assembly"
        folder.mkdir(exist_ok=True)
        with self.subTest("assembly"):
            base_paths = make_gguf(folder / "base.gguf", donor=False)
            donor_path = make_gguf(folder / "donor.gguf", donor=True)[0]
            base = [gguf.GGUFReader(path) for path in base_paths]
            donor = gguf.GGUFReader(donor_path)
            report = audit(base, donor)
            self.assertEqual(report["transplanted_tensors"], 146)
            output_paths = write_output(base, donor, folder / "hybrid.gguf", variant="IQ3_S")
            self.assertEqual(len(output_paths), 2)
            result = [gguf.GGUFReader(path) for path in output_paths]
            tensors = {t.name: t for reader in result for t in reader.tensors}
            self.assertEqual(len(tensors), 148)
            self.assertTrue(np.all(tensors["blk.0.ffn_down_exps.weight"].data == 2))
            self.assertTrue(np.all(tensors["untouched.weight"].data == 3))
            self.assertEqual(tensors["untouched.weight"].tensor_type, gguf.GGMLQuantizationType.Q2_0)
            self.assertTrue(np.all(tensors["per_layer_token_embd.weight"].data == 4))
            self.assertEqual(result[0].get_field("general.license").contents(), "Qwen Community License 1.0")
            self.assertIn("IQ3_S", result[0].get_field("general.name").contents())
            self.assertEqual(verify(base, donor, result)["verified_tensors"], 148)


if __name__ == "__main__":
    unittest.main()
