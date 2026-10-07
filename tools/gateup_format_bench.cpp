// Bounded CPU gate/up packing and routed-kernel preflight. No model is loaded.
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"
#include "ggml-cpu.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <random>
#include <string>
#include <vector>

constexpr int INPUT = 2560, OUTPUT = 640, EXPERTS = 512, ROUTES = 10;

int main(int argc, char ** argv) {
    // type, iterations, native|preferred, fixture directory
    if (argc != 5) return 2;
    const auto type = static_cast<ggml_type>(std::atoi(argv[1]));
    const int iterations = std::atoi(argv[2]);
    const std::string layout = argv[3], folder = argv[4];
    if (iterations < 10 || iterations > 100 || (layout != "native" && layout != "preferred")) return 2;
    const auto traits = ggml_get_type_traits(type);
    if (!traits || !traits->to_float || INPUT % traits->blck_size) return 3;
    const size_t row_bytes = ggml_row_size(type, INPUT), expert_bytes = row_bytes * OUTPUT;
    const size_t matrix_elements = size_t(INPUT) * OUTPUT;
    auto backend = ggml_backend_cpu_init();
    ggml_backend_cpu_set_n_threads(backend, 12);
    const auto device = ggml_backend_get_device(backend);
    ggml_init_params params = {ggml_tensor_overhead() * 32 + ggml_graph_overhead(), nullptr, true};
    auto weights_ctx = ggml_init(params), ctx = ggml_init(params);
    auto w = ggml_new_tensor_3d(weights_ctx, type, INPUT, OUTPUT, EXPERTS);
    auto x = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, INPUT, ROUTES, 1);
    auto ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, ROUTES, 1);
    auto y = ggml_mul_mat_id(ctx, w, x, ids);
    auto buft = ggml_backend_get_default_buffer_type(backend);
    const auto get_extra = reinterpret_cast<ggml_backend_dev_get_extra_bufts_t>(
        ggml_backend_reg_get_proc_address(ggml_backend_cpu_reg(), "ggml_backend_dev_get_extra_bufts"));
    if (layout == "preferred" && get_extra) {
        for (auto options = get_extra(device); options && *options; ++options) {
            auto probe = ggml_backend_buft_alloc_buffer(*options, 0);
            w->buffer = probe; x->buffer = probe; ids->buffer = probe;
            const bool supported = ggml_backend_dev_supports_op(device, y);
            w->buffer = nullptr; x->buffer = nullptr; ids->buffer = nullptr;
            ggml_backend_buffer_free(probe);
            if (supported) { buft = *options; break; }
        }
    }
    auto weights = ggml_backend_alloc_ctx_tensors_from_buft(weights_ctx, buft);
    auto operands = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!weights || !operands) return 4;
    ggml_backend_buffer_clear(weights, 0);
    // Quantize ten independent synthetic templates, then populate all 512
    // distinct expert slots. Rotating routes exercise the full memory working
    // set without spending time quantizing 512 independent source matrices.
    std::mt19937 rng(1729);
    std::normal_distribution<float> normal(0.0f, 0.035f);
    std::vector<float> source(matrix_elements), decoded(matrix_elements), importance(INPUT, 1.0f);
    std::vector<unsigned char> packed(expert_bytes);
    std::vector<float> all_decoded(matrix_elements * ROUTES);
    const std::string fixture = folder + "/" + traits->type_name + ".selected-experts.bin";
    std::ofstream payload_file(fixture, std::ios::binary | std::ios::trunc);
    if (!payload_file) return 5;
    double source_squared = 0, quant_error_squared = 0;
    const auto packing_start = std::chrono::steady_clock::now();
    for (int expert = 0; expert < ROUTES; ++expert) {
        for (auto & value : source) value = normal(rng);
        const size_t actual = ggml_quantize_chunk(type, source.data(), packed.data(), 0, OUTPUT, INPUT, importance.data());
        if (actual != expert_bytes) return 6;
        for (int row = 0; row < OUTPUT; ++row) {
            traits->to_float(packed.data() + row_bytes * row, decoded.data() + size_t(INPUT) * row, INPUT);
        }
        for (size_t j = 0; j < matrix_elements; ++j) {
            if (!std::isfinite(decoded[j])) return 7;
            const double delta = double(source[j]) - decoded[j];
            source_squared += double(source[j]) * source[j];
            quant_error_squared += delta * delta;
        }
        std::copy(decoded.begin(), decoded.end(), all_decoded.begin() + matrix_elements * expert);
        for (int slot = expert; slot < EXPERTS; slot += ROUTES) {
            ggml_backend_tensor_set(w, packed.data(), expert_bytes * slot, expert_bytes);
        }
        payload_file.write(reinterpret_cast<const char *>(packed.data()), packed.size());
    }
    payload_file.close();
    const double packing_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - packing_start).count();
    std::normal_distribution<float> activation(0, 1);
    std::vector<float> input(INPUT * ROUTES), result(OUTPUT * ROUTES), reference(OUTPUT * ROUTES);
    for (auto & value : input) value = activation(rng);
    ggml_backend_tensor_set(x, input.data(), 0, input.size() * sizeof(float));
    auto graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, y);
    auto allocator = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    if (!ggml_gallocr_alloc_graph(allocator, graph)) return 8;
    double maximum_relative_rmse = 0;
    for (int group = 0; group < 5; ++group) {
        double elapsed = 0;
        int32_t routes[ROUTES];
        for (int step = 0; step < iterations + 10; ++step) {
            for (int j = 0; j < ROUTES; ++j) routes[j] = (step * 37 + group * 103 + j * 61) % EXPERTS;
            ggml_backend_tensor_set(ids, routes, 0, sizeof(routes));
            const auto start = std::chrono::steady_clock::now();
            if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS) return 9;
            const auto end = std::chrono::steady_clock::now();
            if (step >= 10) elapsed += std::chrono::duration<double, std::milli>(end - start).count();
        }
        ggml_backend_tensor_get(y, result.data(), 0, result.size() * sizeof(float));
        double reference_squared = 0, error_squared = 0;
        for (int route = 0; route < ROUTES; ++route) {
            for (int row = 0; row < OUTPUT; ++row) {
                const auto row_ptr = all_decoded.data() + matrix_elements * (routes[route] % ROUTES) + size_t(INPUT) * row;
                const auto activation_ptr = input.data() + size_t(INPUT) * route;
                double expected = 0;
                for (int j = 0; j < INPUT; ++j) expected += double(row_ptr[j]) * activation_ptr[j];
                const size_t index = size_t(OUTPUT) * route + row;
                if (!std::isfinite(result[index])) return 10;
                reference[index] = float(expected);
                const double delta = double(result[index]) - expected;
                reference_squared += expected * expected;
                error_squared += delta * delta;
            }
        }
        const double rmse = std::sqrt(error_squared / reference_squared);
        maximum_relative_rmse = std::max(maximum_relative_rmse, rmse);
        std::printf("{\"kind\":\"timing\",\"group\":%d,\"milliseconds_per_routed_matmul\":%.6f,\"iterations\":%d,\"threads\":12,\"experts_per_token\":10,\"shape\":[2560,640,512],\"synthetic_templates\":10,\"populated_experts\":512,\"layout\":\"%s\",\"buffer\":\"%s\",\"relative_rmse_vs_decoded_float_matmul\":%.9f}\n",
            group, elapsed / iterations, iterations, layout.c_str(), ggml_backend_buft_name(buft), rmse);
        std::fflush(stdout);
    }
    std::printf("{\"kind\":\"packing\",\"format\":\"%s\",\"expert_bytes\":%zu,\"tensor_bytes\":%zu,\"selected_payload_bytes\":%zu,\"packing_seconds\":%.6f,\"weight_relative_rmse\":%.9f,\"maximum_kernel_relative_rmse\":%.9f,\"kernel_tolerance\":0.02,\"passed\":%s}\n",
        traits->type_name, expert_bytes, ggml_nbytes(w), expert_bytes * ROUTES, packing_seconds,
        std::sqrt(quant_error_squared / source_squared), maximum_relative_rmse,
        maximum_relative_rmse <= 0.02 ? "true" : "false");
    ggml_gallocr_free(allocator);
    ggml_backend_buffer_free(operands);
    ggml_backend_buffer_free(weights);
    ggml_free(ctx); ggml_free(weights_ctx); ggml_backend_free(backend);
    ggml_quantize_free();
    return maximum_relative_rmse <= 0.02 ? 0 : 11;
}
