// Bounded token-level capture for an adapted gate/up MoE-output-loss pilot.
// Disabled unless GATEUP_CAPTURE_DIR is set. Existing imatrix capture is unchanged.
// GUPACT01: 64-byte LE header, then fixed-size records; all activations are F32.
// Header: magic[8], u32(layer,width,experts,topk,cap,stored),
//         u64(seen,seed,batches,reserved=0).
// Record: u64(token_ordinal,batch_ordinal), u32(row_in_batch,reserved=0),
//         F32 x[width], I32 ids[topk], F32 final_weights[topk], F32 moe_out[width].
// moe_out sums the actual routed, weighted expert outputs, before the shared expert
// and HC scatter. Training/reference weights must share the same frozen down weights.
#pragma once
#include <cstdlib>
#include <cstdint>
#include <cinttypes>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <vector>
#include <map>
#include <set>
#include <string>
#include <sstream>
#include <mutex>
#include <cmath>

struct gateup_record {
    uint64_t token = 0, batch = 0;
    uint32_t row = 0;
    std::vector<float> x, weights, y;
    std::vector<int32_t> ids;
};
struct gateup_layer_capture {
    uint32_t width = 0, experts = 0, topk = 0, cap = 0;
    uint64_t seen = 0, batches = 0;
    bool pending = false, down_verified = false;
    uint32_t pending_rows = 0;
    std::vector<float> pending_x;
    std::vector<int32_t> pending_ids;
    std::vector<gateup_record> records;
};
static std::map<int, gateup_layer_capture> gateup_captures;
static std::mutex gateup_capture_mutex;

static uint64_t gateup_mix(uint64_t x) {
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}
static uint64_t gateup_seed() {
    const char * value = std::getenv("GATEUP_CAPTURE_SEED");
    return value ? std::strtoull(value, nullptr, 10) : 1729;
}
static bool gateup_enabled() {
    const char * value = std::getenv("GATEUP_CAPTURE_DIR");
    return value && *value;
}
static bool gateup_selected(int layer) {
    const char * value = std::getenv("GATEUP_CAPTURE_LAYERS");
    if (!value || !*value) return layer == 0 || layer == 23 || layer == 47;
    std::istringstream input(value);
    std::string item;
    while (std::getline(input, item, ',')) {
        if (std::stoi(item) == layer) return true;
    }
    return false;
}
static int gateup_name_layer(const char * name, const char * prefix) {
    const size_t n = std::strlen(prefix);
    if (std::strncmp(name, prefix, n) != 0) return -1;
    char * end = nullptr;
    const long layer = std::strtol(name + n, &end, 10);
    return end && !*end && layer >= 0 && layer < 48 ? (int) layer : -1;
}
static std::string gateup_weight_name(const char * name) {
    const char * start = std::strchr(name, '#');
    start = start ? start + 1 : name;
    const char * end = std::strchr(start, '#');
    return end ? std::string(start, end - start) : std::string(start);
}
static bool gateup_is_input(const ggml_tensor * t, int & layer) {
    if (t->op != GGML_OP_MUL_MAT_ID || !t->src[0]) return false;
    const auto name = gateup_weight_name(t->src[0]->name);
    // Capture the gate path once, or the fused gate/up path once. The up path
    // has identical x/ids, but collecting both would double count all tokens.
    const auto suffix = name.find(".ffn_gate_exps.weight");
    const auto fused = name.find(".ffn_gate_up_exps.weight");
    if (name.rfind("blk.", 0) != 0 || (suffix == std::string::npos && fused == std::string::npos)) return false;
    const auto dot = name.find('.', 4);
    if (dot == std::string::npos) return false;
    layer = std::stoi(name.substr(4, dot - 4));
    return gateup_selected(layer);
}
static bool gateup_is_down(const ggml_tensor * t, int & layer) {
    if (t->op != GGML_OP_MUL_MAT_ID || !t->src[0]) return false;
    const auto name = gateup_weight_name(t->src[0]->name);
    if (name.rfind("blk.", 0) != 0 || name.find(".ffn_down_exps.weight") == std::string::npos) return false;
    const auto dot = name.find('.', 4);
    if (dot == std::string::npos) return false;
    layer = std::stoi(name.substr(4, dot - 4));
    return gateup_selected(layer);
}
static bool gateup_wants(const ggml_tensor * t) {
    if (!gateup_enabled()) return false;
    int layer = -1;
    if (gateup_is_input(t, layer)) return true;
    if (gateup_is_down(t, layer)) return true;
    layer = gateup_name_layer(t->name, "ffn_moe_weighted-");
    return layer >= 0 && gateup_selected(layer);
}
static std::vector<uint8_t> gateup_read_tensor(const ggml_tensor * t) {
    const size_t size = ggml_nbytes(t);
    GGML_ASSERT(size > 0);
    std::vector<uint8_t> bytes(size);
    ggml_backend_tensor_get(t, bytes.data(), 0, size);
    return bytes;
}
template <typename T>
static T gateup_element(const ggml_tensor * t, const std::vector<uint8_t> & data,
                        int64_t i0, int64_t i1 = 0, int64_t i2 = 0) {
    GGML_ASSERT(i0 >= 0 && i0 < t->ne[0] && i1 >= 0 && i1 < t->ne[1] && i2 >= 0 && i2 < t->ne[2]);
    const size_t offset = i0 * t->nb[0] + i1 * t->nb[1] + i2 * t->nb[2];
    GGML_ASSERT(offset + sizeof(T) <= data.size());
    T result;
    std::memcpy(&result, data.data() + offset, sizeof(T));
    return result;
}
static void gateup_observe(ggml_tensor * t) {
    if (!gateup_enabled()) return;
    std::lock_guard<std::mutex> guard(gateup_capture_mutex);
    int layer = -1;
    if (gateup_is_input(t, layer)) {
        const ggml_tensor * x = t->src[1], * ids = t->src[2], * w = t->src[0];
        GGML_ASSERT(x && ids && x->type == GGML_TYPE_F32 && ids->type == GGML_TYPE_I32);
        GGML_ASSERT(x->ne[0] == 2560 && x->ne[1] == 1 && x->ne[3] == 1);
        GGML_ASSERT(w->ne[2] == 512 && ids->ne[0] == 10 && ids->ne[1] == x->ne[2] && ids->ne[2] == 1);
        GGML_ASSERT(x->ne[2] > 0 && x->ne[2] <= 2048);
        auto & state = gateup_captures[layer];
        // Pending input must match exactly one weighted output before another
        // microbatch arrives. Fail instead of silently pairing unrelated tokens.
        GGML_ASSERT(!state.pending);
        if (!state.width) {
            const char * cap = std::getenv("GATEUP_CAPTURE_CAP");
            state.cap = cap ? std::strtoul(cap, nullptr, 10) : 2048;
            GGML_ASSERT(state.cap > 0 && state.cap <= 8192);
            state.width = 2560; state.experts = 512; state.topk = 10;
            state.records.reserve(state.cap);
        }
        const auto xbytes = gateup_read_tensor(x), idbytes = gateup_read_tensor(ids);
        state.pending_rows = (uint32_t) x->ne[2];
        state.pending_x.resize((size_t) state.pending_rows * state.width);
        state.pending_ids.resize((size_t) state.pending_rows * state.topk);
        for (uint32_t row = 0; row < state.pending_rows; ++row) {
            std::set<int32_t> unique;
            for (uint32_t j = 0; j < state.width; ++j) {
                const float value = gateup_element<float>(x, xbytes, j, 0, row);
                GGML_ASSERT(std::isfinite(value));
                state.pending_x[(size_t) row * state.width + j] = value;
            }
            for (uint32_t k = 0; k < state.topk; ++k) {
                const int32_t value = gateup_element<int32_t>(ids, idbytes, k, row);
                GGML_ASSERT(value >= 0 && value < (int32_t) state.experts && unique.insert(value).second);
                state.pending_ids[(size_t) row * state.topk + k] = value;
            }
        }
        state.pending = true;
        state.down_verified = false;
        return;
    }
    if (gateup_is_down(t, layer)) {
        auto & state = gateup_captures[layer];
        const ggml_tensor * ids = t->src[2];
        GGML_ASSERT(state.pending && !state.down_verified && ids && ids->type == GGML_TYPE_I32);
        GGML_ASSERT(t->ne[0] == state.width && t->ne[1] == state.topk && t->ne[2] == state.pending_rows);
        GGML_ASSERT(ids->ne[0] == state.topk && ids->ne[1] == state.pending_rows);
        const auto idbytes = gateup_read_tensor(ids);
        for (uint32_t row = 0; row < state.pending_rows; ++row) {
            for (uint32_t k = 0; k < state.topk; ++k) {
                GGML_ASSERT(gateup_element<int32_t>(ids, idbytes, k, row) == state.pending_ids[(size_t) row * state.topk + k]);
            }
        }
        state.down_verified = true;
        return;
    }
    layer = gateup_name_layer(t->name, "ffn_moe_weighted-");
    if (layer < 0 || !gateup_selected(layer)) return;
    auto & state = gateup_captures[layer];
    GGML_ASSERT(state.pending && state.down_verified && t->op == GGML_OP_MUL && t->type == GGML_TYPE_F32);
    const ggml_tensor * down = t->src[0], * weights = t->src[1];
    // The scheduler may replace down with a CPU-to-GPU copy that has no op/ids.
    // IDs were verified at the down MUL_MAT_ID while its routing source was live.
    GGML_ASSERT(down && weights && weights->type == GGML_TYPE_F32);
    GGML_ASSERT(t->ne[0] == state.width && t->ne[1] == state.topk && t->ne[2] == state.pending_rows && t->ne[3] == 1);
    GGML_ASSERT(weights->ne[0] == 1 && weights->ne[1] == state.topk && weights->ne[2] == state.pending_rows);
    const auto pbytes = gateup_read_tensor(weights), ybytes = gateup_read_tensor(t);
    ++state.batches;
    for (uint32_t row = 0; row < state.pending_rows; ++row) {
        // Verify grouping even for samples not selected into the reservoir.
        float total = 0.0f;
        for (uint32_t k = 0; k < state.topk; ++k) {
            const float value = gateup_element<float>(weights, pbytes, 0, k, row);
            GGML_ASSERT(std::isfinite(value) && value >= 0.0f);
            total += value;
        }
        // The graph may multiply normalized top-k weights by its configured
        // expert_weights_scale. Save the actual final weights, never guess it.
        GGML_ASSERT(std::isfinite(total) && total > 0.0f && total < 64.0f);
        const uint64_t seen = ++state.seen;
        const uint64_t slot = seen <= state.cap ? seen - 1 : gateup_mix(gateup_seed() ^ gateup_mix(layer) ^ seen) % seen;
        if (slot >= state.cap) continue;
        gateup_record record;
        record.token = seen; record.batch = state.batches; record.row = row;
        record.x.assign(state.pending_x.begin() + (size_t) row * state.width,
                        state.pending_x.begin() + (size_t) (row + 1) * state.width);
        record.ids.assign(state.pending_ids.begin() + (size_t) row * state.topk,
                          state.pending_ids.begin() + (size_t) (row + 1) * state.topk);
        record.weights.resize(state.topk); record.y.resize(state.width, 0.0f);
        for (uint32_t k = 0; k < state.topk; ++k) {
            record.weights[k] = gateup_element<float>(weights, pbytes, 0, k, row);
            for (uint32_t j = 0; j < state.width; ++j) {
                const float value = gateup_element<float>(t, ybytes, j, k, row);
                GGML_ASSERT(std::isfinite(value));
                record.y[j] += value;
            }
        }
        for (const float value : record.y) GGML_ASSERT(std::isfinite(value));
        if (state.records.size() < state.cap) state.records.push_back(std::move(record));
        else state.records[slot] = std::move(record);
    }
    state.pending = false;
    state.down_verified = false;
    state.pending_rows = 0;
    state.pending_x.clear(); state.pending_ids.clear();
}
static void gateup_save() {
    if (!gateup_enabled() || gateup_captures.empty()) return;
    std::lock_guard<std::mutex> guard(gateup_capture_mutex);
    const auto directory = std::filesystem::path(std::getenv("GATEUP_CAPTURE_DIR"));
    std::filesystem::create_directories(directory);
    static_assert(sizeof(float) == 4 && sizeof(uint64_t) == 8 && sizeof(uint32_t) == 4, "capture scalar layout");
    for (const auto & entry : gateup_captures) {
        const auto & state = entry.second;
        // imatrix can checkpoint from inside a gate/down callback. Only fully
        // committed records are written; pending microbatch state stays in RAM.
        const auto path = directory / ("layer" + std::to_string(entry.first) + ".gup");
        const auto temporary = path.string() + ".tmp";
        std::ofstream out(temporary, std::ios::binary | std::ios::trunc);
        const uint32_t dims[6] = {(uint32_t) entry.first, state.width, state.experts, state.topk, state.cap, (uint32_t) state.records.size()};
        const uint64_t scalars[4] = {state.seen, gateup_seed(), state.batches, 0};
        out.write("GUPACT01", 8);
        out.write((const char *) dims, sizeof(dims)); out.write((const char *) scalars, sizeof(scalars));
        for (const auto & record : state.records) {
            const uint64_t ordinals[2] = {record.token, record.batch};
            const uint32_t local[2] = {record.row, 0};
            out.write((const char *) ordinals, sizeof(ordinals)); out.write((const char *) local, sizeof(local));
            out.write((const char *) record.x.data(), record.x.size() * sizeof(float));
            out.write((const char *) record.ids.data(), record.ids.size() * sizeof(int32_t));
            out.write((const char *) record.weights.data(), record.weights.size() * sizeof(float));
            out.write((const char *) record.y.data(), record.y.size() * sizeof(float));
        }
        out.flush(); GGML_ASSERT(out.good()); out.close();
        if (std::filesystem::exists(path)) std::filesystem::remove(path);
        std::filesystem::rename(temporary, path);
        LOG_INF("GATEUP: layer %d: saved %zu / %" PRIu64 " token records (%" PRIu64 " matched microbatches)\n",
                entry.first, state.records.size(), state.seen, state.batches);
    }
}
static void gateup_assert_complete() {
    if (!gateup_enabled()) return;
    GGML_ASSERT(!gateup_captures.empty());
    for (int layer = 0; layer < 48; ++layer) {
        if (gateup_selected(layer)) GGML_ASSERT(gateup_captures.find(layer) != gateup_captures.end());
    }
    for (const auto & entry : gateup_captures) {
        GGML_ASSERT(!entry.second.pending && entry.second.seen > 0 && !entry.second.records.empty());
    }
}
