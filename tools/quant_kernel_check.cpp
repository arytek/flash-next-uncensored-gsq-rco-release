// Check custom packed weights against the actual CPU and CUDA matmul kernels.
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"
#include "ggml-cpu.h"
#include "ggml-cuda.h"
#include <fstream>
#include <vector>
#include <string>
#include <cstdio>

int main(int argc, char ** argv) {
    if (argc != 8) return 2;
    const ggml_type type = (ggml_type) std::stoi(argv[1]);
    const int rows = std::stoi(argv[3]), tokens = std::stoi(argv[5]);
    ggml_backend_t backend = std::string(argv[7]) == "cuda" ? ggml_backend_cuda_init(0) : ggml_backend_cpu_init();
    if (!backend) return 3;
    ggml_init_params params = {ggml_tensor_overhead() * 16 + ggml_graph_overhead(), nullptr, true};
    auto * ctx = ggml_init(params);
    auto * w = ggml_new_tensor_2d(ctx, type, 640, rows);
    auto * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 640, tokens);
    auto * weights = ggml_backend_alloc_ctx_tensors(ctx, backend);
    std::vector<char> packed(ggml_nbytes(w));
    std::vector<float> inputs(640 * tokens);
    std::ifstream wf(argv[2], std::ios::binary), xf(argv[4], std::ios::binary);
    wf.read(packed.data(), packed.size()); xf.read((char *) inputs.data(), inputs.size() * sizeof(float));
    if (!wf || !xf) return 4;
    ggml_backend_tensor_set(w, packed.data(), 0, packed.size());
    ggml_backend_tensor_set(x, inputs.data(), 0, inputs.size() * sizeof(float));
    auto * y = ggml_mul_mat(ctx, w, x);
    auto * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, y);
    auto allocator = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    if (!ggml_gallocr_alloc_graph(allocator, graph)) return 5;
    if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS) return 6;
    std::vector<float> result(rows * tokens);
    ggml_backend_tensor_get(y, result.data(), 0, result.size() * sizeof(float));
    std::ofstream output(argv[6], std::ios::binary);
    output.write((char *) result.data(), result.size() * sizeof(float));
    ggml_gallocr_free(allocator); ggml_backend_buffer_free(weights);
    ggml_free(ctx); ggml_backend_free(backend);
    return output.good() ? 0 : 7;
}
