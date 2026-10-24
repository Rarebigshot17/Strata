// src/glm/expert_cache.cu - the routed experts most used, held in VRAM (expert_cache.hpp).
#include "strata/glm/expert_cache.hpp"

#include "ggml.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdlib>
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
    verify_ = std::getenv("STRATA_GLM_VERIFY") && std::atoi(std::getenv("STRATA_GLM_VERIFY")) > 0;
    sync_swaps_ = std::getenv("STRATA_GLM_SYNC_SWAPS") && std::atoi(std::getenv("STRATA_GLM_SYNC_SWAPS")) > 0;
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
    copier_ = std::thread([this] { copier(); });
}

ExpertCache::~ExpertCache() {
    {
        std::lock_guard<std::mutex> lk(mu_);
        stop_ = true;
    }
    cv_todo_.notify_all();
    if (copier_.joinable()) copier_.join();
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
    int n = (int) std::min<size_t>(budget / per_round, (size_t) c.n_expert);
    if (opt_.max_slots > 0) n = std::min(n, opt_.max_slots);
    if (n <= 0) return 0;
    size_t used = 0;
    for (int64_t il : mine) {
        ExpertSlots& s = L_[(size_t) il];
        s.dev = dev;
        s.n_slots = n;
        s.n_own = n;
        ok(cudaMalloc(&s.d, (size_t) n * s.blob), "cudaMalloc for the expert cache");
        s.expert_in.assign((size_t) n, -1);
        used += (size_t) n * s.blob;
    }
    return used;
}

size_t ExpertCache::add_loans(int dev, const std::vector<int64_t>& layers,
                              const std::vector<std::pair<uint8_t*, size_t>>& chunks) {
    if (opt_.disabled) return 0;
    const Config& c = m_.config();
    std::vector<int64_t> mine;
    size_t blob = 0;
    for (int64_t il : layers) {
        const ExpertSlots& s = L_[(size_t) il];
        if (c.is_dense(il) || !s.cacheable) continue;
        mine.push_back(il);
        blob = std::max(blob, s.blob);
    }
    if (mine.empty() || blob == 0) return 0;
    // the blobs each buffer holds, one after another from its start (cudaMalloc's alignment, blobs of 256s)
    std::vector<uint8_t*> at;
    for (const auto& [p, n] : chunks)
        for (size_t o = 0; p && o + blob <= n; o += blob) at.push_back(p + o);
    int per = (int) (at.size() / mine.size());
    size_t used = 0, next = 0;
    for (int64_t il : mine) {
        ExpertSlots& s = L_[(size_t) il];
        const int k = std::min<int>(per, (int) c.n_expert - s.n_own);
        if (k <= 0) continue;
        if (s.n_slots == 0) {   // no slots of its own (allocate found no room): lent ones only
            s.dev = dev;
            s.n_own = 0;
            s.expert_in.clear();
        }
        for (int i = 0; i < k; ++i) s.loan.push_back(at[next++]);
        s.n_slots = s.n_own + (int) s.loan.size();
        s.expert_in.resize((size_t) s.n_slots, -1);
        lent_slots_ += k;
        used += (size_t) k * s.blob;
    }
    return used;
}

void ExpertCache::reclaim_loans() {
    // copies queued into lent slots are dropped rather than waited for (after a prompt the misses queue up to every
    // lent slot's fill); the one in flight, and the copies into the cache's own slots, land first
    {
        std::lock_guard<std::mutex> lk(mu_);
        std::deque<Copy> keep;
        for (const Copy& j : todo_) {
            ExpertSlots& s = L_[(size_t) j.il];
            if (j.slot < s.n_own) {
                keep.push_back(j);
                continue;
            }
            s.slot_of[(size_t) j.e] = -1;
            s.expert_in[(size_t) j.slot] = -1;
        }
        todo_.swap(keep);
    }
    drain();
    for (auto& s : L_)
        for (int i = s.n_own; i < s.n_slots; ++i) {
            const int ex = s.expert_in[(size_t) i];
            if (ex >= 0) {
                s.slot_of[(size_t) ex] = -1;
                ++lent_dropped_;
            }
            s.expert_in[(size_t) i] = kLent;
        }
}

void ExpertCache::return_loans() {
    for (auto& s : L_)
        for (int i = s.n_own; i < s.n_slots; ++i)
            if (s.expert_in[(size_t) i] == kLent) s.expert_in[(size_t) i] = -1;
}

void ExpertCache::copy_expert(int64_t il, int e, int slot, void* stream) {
    const ExpertSlots& s = L_[(size_t) il];
    const LayerWeights& w = m_.layer(il);
    ok(cudaSetDevice(s.dev), "cudaSetDevice");
    uint8_t* dst = s.at(slot);
    const Tensor g = w.gate_exps.slice2(e), u = w.up_exps.slice2(e), d = w.down_exps.slice2(e);
    const cudaStream_t st = (cudaStream_t) stream;
    ok(cudaMemcpyAsync(dst, g.data, s.up_off, cudaMemcpyHostToDevice, st), "expert copy");
    ok(cudaMemcpyAsync(dst + s.up_off, u.data, s.down_off - s.up_off, cudaMemcpyHostToDevice, st), "expert copy");
    ok(cudaMemcpyAsync(dst + s.down_off, d.data, s.down.row_bytes * (size_t) s.down.n_out, cudaMemcpyHostToDevice,
                       st), "expert copy");
    ok(cudaStreamSynchronize(st), "expert copy");
}

void ExpertCache::load(int64_t il, int e, int slot) {
    ExpertSlots& s = L_[(size_t) il];
    const int old = s.expert_in[(size_t) slot];
    if (old >= 0) s.slot_of[(size_t) old] = -1;
    copy_expert(il, e, slot, nullptr);   // at start (warm_start), before any kernel runs
    s.expert_in[(size_t) slot] = e;
    s.slot_of[(size_t) e] = slot;
}

bool ExpertCache::check_slot(int64_t il, int slot, int e, const char* what) const {
    const ExpertSlots& s = L_[(size_t) il];
    const LayerWeights& w = m_.layer(il);
    std::vector<uint8_t> got(s.blob);
    ok(cudaSetDevice(s.dev), "cudaSetDevice");
    ok(cudaMemcpy(got.data(), s.at(slot), s.blob, cudaMemcpyDeviceToHost), "slot check");
    const Tensor g = w.gate_exps.slice2(e), u = w.up_exps.slice2(e), d = w.down_exps.slice2(e);
    const struct {
        const char* name;
        const void* src;
        size_t off, n;
    } parts[3] = {{"gate", g.data, 0, s.up_off},
                  {"up", u.data, s.up_off, s.down_off - s.up_off},
                  {"down", d.data, s.down_off, s.down.row_bytes * (size_t) s.down.n_out}};
    for (const auto& pt : parts) {
        const uint8_t* a = got.data() + pt.off;
        const uint8_t* b = static_cast<const uint8_t*>(pt.src);
        if (std::memcmp(a, b, pt.n) == 0) continue;
        size_t i = 0;
        while (i < pt.n && a[i] == b[i]) ++i;
        size_t bad = 0;
        for (size_t j = i; j < pt.n; ++j) bad += a[j] != b[j];
        std::fprintf(stderr, "expert check: %s: layer %lld expert %d slot %d on GPU %d: its %s differs from RAM at "
                             "byte %zu of %zu (%zu bytes differ)\n", what, (long long) il, e, slot, s.dev, pt.name,
                     i, pt.n, bad);
        return false;
    }
    return true;
}

void ExpertCache::copier() {
    std::vector<cudaStream_t> streams;   // per device, made here
    for (;;) {
        Copy job;
        {
            std::unique_lock<std::mutex> lk(mu_);
            cv_todo_.wait(lk, [&] { return stop_ || !todo_.empty(); });
            if (stop_) return;
            job = todo_.front();
            todo_.pop_front();
            ++busy_;
        }
        const int dev = L_[(size_t) job.il].dev;
        bool copied = false;
        try {
            if ((int) streams.size() <= dev) streams.resize((size_t) dev + 1, nullptr);
            ok(cudaSetDevice(dev), "cudaSetDevice");
            if (!streams[(size_t) dev])
                ok(cudaStreamCreateWithFlags(&streams[(size_t) dev], cudaStreamNonBlocking), "cudaStreamCreate");
            copy_expert(job.il, job.e, job.slot, streams[(size_t) dev]);
            copied = true;
            if (verify_) check_slot(job.il, job.slot, job.e, "a swap's copy");
        } catch (const std::exception& ex) {
            std::fprintf(stderr, "expert cache: copying expert %d of layer %lld failed (%s); its slot stays empty\n",
                         job.e, (long long) job.il, ex.what());
        }
        std::lock_guard<std::mutex> lk(mu_);
        if (!copied) job.e = -1;
        done_.push_back(job);
        --busy_;
        if (todo_.empty() && busy_ == 0) cv_idle_.notify_all();
    }
}

void ExpertCache::apply_done() {
    std::vector<Copy> done;
    {
        std::lock_guard<std::mutex> lk(mu_);
        done.swap(done_);
    }
    for (const Copy& c : done) {
        ExpertSlots& s = L_[(size_t) c.il];
        s.expert_in[(size_t) c.slot] = c.e;
        if (c.e >= 0) s.slot_of[(size_t) c.e] = c.slot;
    }
}

void ExpertCache::drain() {
    {
        std::unique_lock<std::mutex> lk(mu_);
        cv_idle_.wait(lk, [&] { return todo_.empty() && busy_ == 0; });
    }
    apply_done();
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
    return s.n_slots && s.slot_of[(size_t) e] >= 0 ? s.slot_of[(size_t) e] : -1;
}

void ExpertCache::matrices(int64_t il, int slot, gpu::DevMatrix& gate, gpu::DevMatrix& up,
                           gpu::DevMatrix& down) const {
    const ExpertSlots& s = L_[(size_t) il];
    uint8_t* base = s.at(slot);
    gate = s.gate;
    gate.w = base;
    up = s.up;
    up.w = base + s.up_off;
    down = s.down;
    down.w = base + s.down_off;
}

int ExpertCache::claim_free(int64_t il, int e) {
    ExpertSlots& s = L_[(size_t) il];
    if (s.n_slots == 0 || s.slot_of[(size_t) e] != -1) return -1;
    for (int i = 0; i < s.n_slots; ++i)
        if (s.expert_in[(size_t) i] == -1) {
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
    apply_done();
    const Config& c = m_.config();
    if (verify_ && tick_ % 512 == 0) {
        // every cached slot against RAM (the slots being copied into are skipped)
        int64_t n = 0, bad = 0;
        for (int64_t il = 0; il < c.n_layer; ++il) {
            const ExpertSlots& s = L_[(size_t) il];
            for (int i = 0; i < s.n_slots; ++i) {
                const int ex = s.expert_in[(size_t) i];
                if (ex < 0) continue;
                ++n;
                if (!check_slot(il, i, ex, "the cache") && ++bad >= 20) break;
            }
            if (bad >= 20) break;
        }
        std::fprintf(stderr, "expert check: token %lld: %lld cached slots checked, %lld differ from RAM%s\n",
                     (long long) tick_, (long long) n, (long long) bad, bad >= 20 ? " (stopped at 20)" : "");
    }
    for (auto& row : score_)
        for (auto& v : row) v *= (float) opt_.decay;
    // filling a free slot costs one copy and displaces nothing, so the cache fills four times faster than it swaps
    // the starting layer rotates, so a cap reached early in the loop does not always favour the first layers
    int swaps = 0, fills = 0;
    // copies still queued or running count against this token's swaps, so the copy thread never falls behind by
    // more than a token's worth (a backlog leaves evicted slots empty while it waits)
    {
        std::lock_guard<std::mutex> lk(mu_);
        swaps = std::min<int>((int) todo_.size() + busy_, opt_.max_swaps_per_token);
    }
    const int64_t first = tick_++ % c.n_layer;
    for (int64_t n = 0; n < c.n_layer; ++n) {
        const int64_t il = (first + n) % c.n_layer;
        ExpertSlots& s = L_[(size_t) il];
        auto& miss = missed_[(size_t) il];
        for (int e : miss) {
            if (swaps >= opt_.max_swaps_per_token && fills >= 4 * opt_.max_swaps_per_token) break;
            if (s.slot_of[(size_t) e] != -1) continue;   // cached, or being copied
            const auto& sc = score_[(size_t) il];
            int victim = -1;
            float vscore = 0.0f;
            for (int i = 0; i < s.n_slots; ++i) {
                const int ex = s.expert_in[(size_t) i];
                if (ex == kLoading || ex == kLent) continue;
                if (ex < 0) { victim = i; vscore = -1.0f; break; }
                if (victim < 0 || sc[(size_t) ex] < vscore) { victim = i; vscore = sc[(size_t) ex]; }
            }
            if (victim < 0) break;
            if (vscore < 0.0f) {
                if (fills >= 4 * opt_.max_swaps_per_token) continue;
                ++fills;
                ++fills_queued_;
            } else {
                if (swaps >= opt_.max_swaps_per_token || !(sc[(size_t) e] > 1.2f * vscore + 0.5f)) continue;
                ++swaps;
                ++swaps_queued_;
            }
            const int old = s.expert_in[(size_t) victim];
            if (old >= 0) s.slot_of[(size_t) old] = -1;   // no kernel reads it now: the token's work is done
            s.expert_in[(size_t) victim] = kLoading;
            s.slot_of[(size_t) e] = kLoading;
            {
                std::lock_guard<std::mutex> lk(mu_);
                todo_.push_back({il, e, victim});
            }
            cv_todo_.notify_one();
        }
        miss.clear();
    }
    if (sync_swaps_) {
        // the copies queued now land before the next token runs (a test of whether copies overlapping work matter)
        {
            std::unique_lock<std::mutex> lk(mu_);
            cv_idle_.wait(lk, [&] { return todo_.empty() && busy_ == 0; });
        }
        apply_done();
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
    size_t lent = 0;
    for (const auto& s : L_) lent += s.loan.size() * s.blob;
    char buf[300];
    std::snprintf(buf, sizeof buf, "expert cache: %lld slots over %lld layers (%.2f GB)%s", (long long) slots,
                  (long long) layers, bytes / 1e9, loaded_profile_ ? ", filled from the profile" : "");
    std::string out = buf;
    if (lent_slots_ > 0) {
        std::snprintf(buf, sizeof buf, "; %lld of them (%.2f GB) in the prompt path's buffers between prompts",
                      (long long) lent_slots_, lent / 1e9);
        out += buf;
    }
    return out;
}

}  // namespace strata::glm
