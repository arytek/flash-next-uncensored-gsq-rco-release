"""Install a disabled-by-default token-grouped gate/up capture in local imatrix.

This adds capture only. It implements neither RCO nor a differentiable model.
"""
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
target = ROOT / "third_party/llama.cpp/tools/imatrix/imatrix.cpp"
contents = target.read_text(encoding="utf-8")
if '#include "gateup_capture.h"' not in contents:
    needle = '#include "activation_reservoir.h"'
    if contents.count(needle) != 1:
        raise ValueError("Expected existing activation collector include")
    contents = contents.replace(needle, needle + '\n#include "gateup_capture.h"', 1)
if "gateup_wants(t)" not in contents:
    needle = 'bool IMatrixCollector::collect_imatrix(struct ggml_tensor * t, bool ask, void * user_data) {\n    GGML_UNUSED(user_data);'
    if contents.count(needle) != 1:
        raise ValueError("Unexpected imatrix callback signature")
    contents = contents.replace(needle, needle + "\n\n    // Independent capture handles non-matmul output nodes before imatrix's src0 assumptions.\n"
                                "    if (gateup_wants(t)) {\n"
                                "        if (ask) return true;\n"
                                "        gateup_observe(t);\n"
                                "        if (t->op != GGML_OP_MUL_MAT_ID) return true;\n"
                                "    }", 1)
if "    gateup_save();" not in contents:
    needle = "void IMatrixCollector::save_imatrix(int32_t n_chunk) const {"
    if contents.count(needle) != 1:
        raise ValueError("Unexpected imatrix save signature")
    contents = contents.replace(needle, needle + "\n    gateup_save();", 1)
if "    gateup_assert_complete();" not in contents:
    needle = "    g_collector.save_imatrix();"
    position = contents.rfind(needle)
    if position < 0:
        raise ValueError("No final imatrix checkpoint found")
    contents = contents[:position] + "    gateup_assert_complete();\n" + contents[position:]
target.write_text(contents, encoding="utf-8")
shutil.copyfile(ROOT / "tools/gateup_capture.h", target.parent / "gateup_capture.h")
print("Gate/up token capture installed. Rebuild llama-imatrix before using GATEUP_CAPTURE_DIR.")
