"""Reproducibly add bounded expert activation capture to the private llama.cpp checkout."""
from pathlib import Path
import shutil

root = Path(__file__).resolve().parents[1]
target = root / 'third_party/llama.cpp/tools/imatrix/imatrix.cpp'
text = target.read_text(encoding='utf-8')
if '#include "activation_reservoir.h"' not in text:
    text = text.replace('class IMatrixCollector {', '#include "activation_reservoir.h"\n\nclass IMatrixCollector {', 1)
    needle = '                e.counts[ex]++;'
    assert text.count(needle) == 1
    text = text.replace(needle, '                GGML_ASSERT(src1->type == GGML_TYPE_F32);\n'
                        '                rco_capture(wname, n_as, ex, ne0, x);\n' + needle, 1)
    needle = 'void IMatrixCollector::save_imatrix(int32_t n_chunk) const {'
    assert text.count(needle) == 1
    text = text.replace(needle, needle + '\n    rco_save();', 1)
    target.write_text(text, encoding='utf-8')
shutil.copyfile(root / 'tools/activation_reservoir.h', target.parent / 'activation_reservoir.h')
cmake = root / 'third_party/llama.cpp/CMakeLists.txt'
definition = '\nadd_executable(rco-kernel-check "${CMAKE_SOURCE_DIR}/../../tools/quant_kernel_check.cpp")\n'
definition += 'target_link_libraries(rco-kernel-check PRIVATE ggml ggml-base ggml-cpu ggml-cuda)\n'
contents = cmake.read_text(encoding='utf-8')
if 'add_executable(rco-kernel-check' not in contents:
    cmake.write_text(contents + definition, encoding='utf-8')
contents = cmake.read_text(encoding='utf-8')
if 'add_executable(rco-expert-format-bench' not in contents:
    cmake.write_text(contents+'\nadd_executable(rco-expert-format-bench "${CMAKE_SOURCE_DIR}/../../tools/expert_format_bench.cpp")\n'
                     +'target_link_libraries(rco-expert-format-bench PRIVATE ggml ggml-base ggml-cpu)\n',encoding='utf-8')
print('Activation capture installed; rebuild llama-imatrix before use.')
