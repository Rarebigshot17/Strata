// src/glm/expert_cache.cu - the routed experts most used, held in VRAM (expert_cache.hpp).
#include "strata/glm/expert_cache.hpp"

#include "ggml.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <numeric>
#include <stdexcept>

namespace strata::glm {

namespace {

void ok(cudaError_t e, const char* what) {
    if (e != cudaSuccess) throw std::runtime_error(std::string("CUDA ") + what + ": " + cudaGetErrorString(e));
}

constexpr uint32_t kProfileMagic = 0x504d4c47;   // "GLMP"

gpu::DevMatrix shape_of(const Tensor& t) {
    gpu::DevMatrix m;
    m.type = (int) t.type;
    m.path = gpu::path_for((int) t.type);
    m.n_in = t.ne[0];
    m.n_out = t.ne[1];
    m.row_bytes = t.row_bytes();
    return m;
}

}  // namespace

ExpertCache::ExpertCache(const Model& model, const ExpertCacheOptions& opt) : m_(model), opt_(opt) {
    const Config& c = m_.config();
    L_.resize((size_t) c.n_layer);
    score_.assign((size_t) c.n_layer, std::vector<float>((size_t) c.n_expert, 0.0f));
    missed_.resize((size_t) c.n_layer);
    for (int64_t il = 0; il < c.n_layer; ++il) {
        ExpertSlots& s = L_[(size_t) il];
        s.slot_of.assign((size_t) c.n_expert, -1);
        if (c.is_dense(il)) continue;
        const LayerWeights& w = m_.layer(il);
        s.gate = shape_of(w.gate_exps);
        s.up = shape_of(w.up_exps);
        s.down = shape_of(w.down_exps);
        s.cacheable = s.gate.path != gpu::DevMatrix::None && s.up.path != gpu::DevMatrix::None &&
                      s.down.path != gpu::DevMatrix::None;
        const size_t gb = s.gate.row_bytes * (size_t) s.gate.n_out;
        const size_t ub = s.up.row_bytes * (size_t) s.up.n_out;
        const size_t db = s.down.row_bytes * (size_t) s.down.n_out;
        s.up_off = gb;
        s.down_off = gb + ub;
        s.blob = (gb + ub + db + 255) / 256 * 256;
    }
    if (!opt_.profile.empty()) {
        std::ifstream in(opt_.profile, std::ios::binary);
        uint32_t hdr[3] = {0, 0, 0};
        if (in && in.read(reinterpret_cast<char*>(hdr), sizeof hdr) && hdr[0] == kProfileMagic &&
            hdr[1] == (uint32_t) c.n_layer && hdr[2] == (uint32_t) c.n_expert) {
            for (auto& row : score_) in.read(reinterpret_cast<char*>(row.data()), (std::streamsize) (row.size() * sizeof(float)));
            loaded_profile_ = (bool) in;
        }
    }
}

size_t ExpertCache::allocate(int dev, const std::vector<int64_t>& layers) {
    if (opt_.disabled) return 0;
    const Config& c = m_.config();
    std::vector<int64_t> mine;
    size_t per_round = 0;
    for (int64_t il : layers) {
        ExpertSlots& s = L_[(size_t) il];
        if (c.is_dense(il)) continue;
        if (!s.cacheable) {
            std::fprintf(stderr, "layer %lld: the experts' types have no GPU kernel, they all run on the CPU\n",
                         (long long) il);
            continue;
        }
        mine.push_back(il);
        per_round += s.blob;
    }
    if (mine.empty()) return 0;
    ok(cudaSetDevice(dev), "cudaSetDevice");
    size_t fr = 0, tot = 0;
    ok(cudaMemGetInfo(&fr, &tot), "cudaMemGetInfo");
    const size_t reserve = (size_t) (opt_.vram_reserve_gb * 1e9);
    const size_t budget = fr > reserve ? fr - reserve : 0;
    const int n = (int) std::min<size_t>(budget / per_round, (size_t) c.n_expert);
    if (n <= 0) return 0;
    size_t used = 0;
    for (int64_t il : mine) {
        ExpertSlots& s = L_[(size_t) il];
        s.dev = dev;
        s.n_slots = n;
        ok(cudaMalloc(&s.d, (size_t) n * s.blob), "cudaMalloc for the expert cache");
        s.expert_in.assign((size_t) n, -1);
        used += (size_t) n * s.blob;
    }
    return used;
}

void ExpertCache::load(int64_t il, int e, int slot) {
    ExpertSlots& s = L_[(size_t) il];
    const LayerWeights& w = m_.layer(il);
    const int old = s.expert_in[(size_t) slot];
    if (old >= 0) s.slot_of[(size_t) old] = -1;
    ok(cudaSetDevice(s.dev), "cudaSetDevice");
    uint8_t* dst = s.d + (size_t) slot * s.blob;
    const Tensor g = w.gate_exps.slice2(e), u = w.up_exps.slice2(e), d = w.down_exps.slice2(e);
    ok(cudaMemcpy(dst, g.data, s.up_off, cudaMemcpyHostToDevice), "expert copy");
    ok(cudaMemcpy(dst + s.up_off, u.data, s.down_off - s.up_off, cudaMemcpyHostToDevice), "expert copy");
    ok(cudaMemcpy(dst + s.down_off, d.data, s.down.row_bytes * (size_t) s.down.n_out, cudaMemcpyHostToDevice),
       "expert copy");
    s.expert_in[(size_t) slot] = e;
    s.slot_of[(size_t) e] = slot;
}

void ExpertCache::warm_start() {
    if (!loaded_profile_) return;
    const Config& c = m_.config();
    for (int64_t il = 0; il < c.n_layer; ++il) {
        ExpertSlots& s = L_[(size_t) il];
        if (s.n_slots == 0) continue;
        std::vector<int> order((size_t) c.n_expert);
        std::iota(order.begin(), order.end(), 0);
        const auto& sc = score_[(size_t) il];
        std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return sc[(size_t) a] > sc[(size_t) b]; });
        for (int i = 0; i < s.n_slots && sc[(size_t) order[(size_t) i]] > 0.0f; ++i) load(il, order[(size_t) i], i);
    }
}

int ExpertCache::slot(int64_t il, int e) const {
    const ExpertSlots& s = L_[(size_t) il];
    return s.n_slots ? s.slot_of[(size_t) e] : -1;
}

void ExpertCache::matrices(int64_t il, int slot, gpu::DevMatrix& gate, gpu::DevMatrix& up,
                           gpu::DevMatrix& down) const {
    const ExpertSlots& s = L_[(size_t) il];
    uint8_t* base = s.d + (size_t) slot * s.blob;
    gate = s.gate;
    gate.w = base;
    up = s.up;
    up.w = base + s.up_off;
    down = s.down;
    down.w = base + s.down_off;
}

int ExpertCache::claim_free(int64_t il, int e) {
    ExpertSlots& s = L_[(size_t) il];
    if (s.n_slots == 0 || s.slot_of[(size_t) e] >= 0) return -1;
    for (int i = 0; i < s.n_slots; ++i)
        if (s.expert_in[(size_t) i] < 0) {
            s.expert_in[(size_t) i] = e;
            s.slot_of[(size_t) e] = i;
            return i;
        }
    return -1;
}

void ExpertCache::record(int64_t il, const int32_t* ids, int k) {
    const ExpertSlots& s = L_[(size_t) il];
    for (int j = 0; j < k; ++j) {
        score_[(size_t) il][(size_t) ids[j]] += 1.0f;
        ++lookups_;
        if (s.n_slots && s.slot_of[(size_t) ids[j]] >= 0) ++hits_;
        else if (s.n_slots) missed_[(size_t) il].push_back(ids[j]);
    }
}

void ExpertCache::maintain() {
    const Config& c = m_.config();
    for (auto& row : score_)
        for (auto& v : row) v *= (float) opt_.decay;
    // filling a free slot costs one copy and displaces nothing, so the cache fills four times faster than it swaps
    // the starting layer rotates, so a cap reached early in the loop does not always favour the first layers
    int swaps = 0, fills = 0;
    const int64_t first = tick_++ % c.n_layer;
    for (int64_t n = 0; n < c.n_layer; ++n) {
        const int64_t il = (first + n) % c.n_layer;
        ExpertSlots& s = L_[(size_t) il];
        auto& miss = missed_[(size_t) il];
        for (int e : miss) {
            if (swaps >= opt_.max_swaps_per_token && fills >= 4 * opt_.max_swaps_per_token) break;
            if (s.slot_of[(size_t) e] >= 0) continue;
            const auto& sc = score_[(size_t) il];
            int victim = -1;
            float vscore = 0.0f;
            for (int i = 0; i < s.n_slots; ++i) {
                const int ex = s.expert_in[(size_t) i];
                if (ex < 0) { victim = i; vscore = -1.0f; break; }
                if (victim < 0 || sc[(size_t) ex] < vscore) { victim = i; vscore = sc[(size_t) ex]; }
            }
            if (victim < 0) break;
            if (vscore < 0.0f) {
                if (fills >= 4 * opt_.max_swaps_per_token) continue;
                ++fills;
            } else {
                if (swaps >= opt_.max_swaps_per_token || !(sc[(size_t) e] > 1.2f * vscore + 0.5f)) continue;
                ++swaps;
            }
            load(il, e, victim);
        }
        miss.clear();
    }
}

void ExpertCache::save_profile() const {
    if (opt_.profile.empty()) return;
    const Config& c = m_.config();
    const std::string tmp = opt_.profile + ".tmp";
    {
        std::ofstream out(tmp, std::ios::binary);
        const uint32_t hdr[3] = {kProfileMagic, (uint32_t) c.n_layer, (uint32_t) c.n_expert};
        out.write(reinterpret_cast<const char*>(hdr), sizeof hdr);
        for (const auto& row : score_)
            out.write(reinterpret_cast<const char*>(row.data()), (std::streamsize) (row.size() * sizeof(float)));
        if (!out) return;
    }
    std::rename(tmp.c_str(), opt_.profile.c_str());
}

int64_t ExpertCache::total_slots() const {
    int64_t n = 0;
    for (const auto& s : L_) n += s.n_slots;
    return n;
}

size_t ExpertCache::total_bytes() const {
    size_t b = 0;
    for (const auto& s : L_) b += (size_t) s.n_slots * s.blob;
    return b;
}

std::string ExpertCache::summary() const {
    int64_t slots = 0, layers = 0;
    size_t bytes = 0;
    for (const auto& s : L_)
        if (s.n_slots) {
            slots += s.n_slots;
            ++layers;
            bytes += (size_t) s.n_slots * s.blob;
        }
    char buf[200];
    std::snprintf(buf, sizeof buf, "expert cache: %lld slots over %lld layers (%.2f GB)%s", (long long) slots,
                  (long long) layers, bytes / 1e9, loaded_profile_ ? ", filled from the profile" : "");
    return buf;
}

}  // namespace strata::glm
