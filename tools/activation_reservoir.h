// Bounded, deterministic routed-input reservoir for the local RCO experiment.
// Persisted little-endian format: RCOACT01, u32 experts/width/cap/reserved,
// u64 seen[experts], then IEEE FP16 vectors[experts, cap, width].
#pragma once
#include <cstdlib>
#include <filesystem>
#include <cstdint>

struct rco_reservoir {
    int experts = 0, width = 0, cap = 0;
    std::vector<uint64_t> seen;
    std::vector<ggml_fp16_t> vectors;
};
static std::map<std::string, rco_reservoir> rco_inputs;
static uint64_t rco_mix(uint64_t x) {
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}
static void rco_capture(const std::string & name, int experts, int ex, int width, const float * x) {
    const char * directory = std::getenv("RCO_CAPTURE_DIR");
    if (!directory || !*directory || name.find("ffn_down_exps.weight") == std::string::npos) return;
    auto & r = rco_inputs[name];
    if (r.seen.empty()) {
        const char * cap = std::getenv("RCO_CAPTURE_CAP");
        r.cap = cap ? std::atoi(cap) : 64;
        GGML_ASSERT(r.cap > 0 && r.cap <= 128 && width == 640 && experts == 512);
        r.experts = experts;
        r.width = width;
        r.seen.resize(experts, 0);
        r.vectors.resize((size_t) experts * r.cap * width, 0);
    }
    uint64_t seed = 1729;
    for (unsigned char c : name) seed = rco_mix(seed ^ c);
    const uint64_t seen = ++r.seen[ex];
    const uint64_t slot = seen <= (uint64_t) r.cap ? seen - 1 : rco_mix(seed ^ rco_mix(ex) ^ seen) % seen;
    if (slot < (uint64_t) r.cap) {
        auto * dst = r.vectors.data() + ((size_t) ex * r.cap + slot) * width;
        for (int j = 0; j < width; ++j) {
            GGML_ASSERT(std::isfinite(x[j]));
            dst[j] = ggml_fp32_to_fp16(x[j]);
        }
    }
}
static void rco_save() {
    const char * directory = std::getenv("RCO_CAPTURE_DIR");
    if (!directory || !*directory || rco_inputs.empty()) return;
    std::filesystem::create_directories(directory);
    for (const auto & entry : rco_inputs) {
        const auto & r = entry.second;
        const auto path = std::filesystem::path(directory) / (entry.first + ".act");
        std::ofstream out(path, std::ios::binary | std::ios::trunc);
        const uint32_t dimensions[4] = {(uint32_t) r.experts, (uint32_t) r.width, (uint32_t) r.cap, 0};
        out.write("RCOACT01", 8);
        out.write((const char *) dimensions, sizeof(dimensions));
        out.write((const char *) r.seen.data(), r.seen.size() * sizeof(uint64_t));
        out.write((const char *) r.vectors.data(), r.vectors.size() * sizeof(ggml_fp16_t));
        out.flush();
        GGML_ASSERT(out.good());
    }
    LOG_INF("RCO: saved routed input reservoirs for %zu tensors\n", rco_inputs.size());
}
