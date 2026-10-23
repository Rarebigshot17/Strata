// src/glm/pinned_experts.cu - the routed experts in page-locked RAM (pinned_experts.hpp).
#include "strata/glm/pinned_experts.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <string>
#include <thread>

#include <sys/mman.h>
#include <unistd.h>

namespace strata::glm {

size_t mem_available() {
    std::ifstream in("/proc/meminfo");
    std::string key, unit;
    unsigned long long kb = 0;
    while (in >> key >> kb) {
        std::getline(in, unit);
        if (key == "MemAvailable:") return (size_t) kb * 1024;
    }
    return 0;
}

PinnedExperts::PinnedExperts(Model& model, int threads, size_t keep_free,
                             const std::function<void(size_t, size_t)>& progress) {
    const Config& c = model.config();
    struct Piece {
        const uint8_t* src;
        uint8_t* dst;
        size_t n;
    };
    struct Move {
        int64_t il;
        uint8_t* dst[3];
    };
    std::vector<Piece> pieces;     // the copies, in 64 MiB pieces
    std::vector<Piece> sources;    // each tensor's whole source range, for its pages to be let go after
    std::vector<Move> moves;
    const size_t piece = (size_t) 64 << 20, align = 4096;
    size_t total = 0;
    for (int64_t il = 0; il < c.n_layer; ++il) {
        if (c.is_dense(il)) continue;
        const LayerWeights& w = model.layer(il);
        const Tensor* ts[3] = {&w.gate_exps, &w.up_exps, &w.down_exps};
        if (!ts[0]->ok() || !ts[1]->ok() || !ts[2]->ok()) continue;
        size_t n[3], off[3], size = 0;
        for (int k = 0; k < 3; ++k) {
            n[k] = ts[k]->row_bytes() * (size_t) ts[k]->rows();
            off[k] = size;
            size += (n[k] + align - 1) / align * align;
        }
        if (!why_.empty()) {
            ++left_;
            continue;
        }
        const size_t avail = mem_available();
        if (avail != 0 && avail < size + keep_free) {
            why_ = "MemAvailable " + std::to_string(avail >> 30) + " GiB would fall below the " +
                   std::to_string(keep_free >> 30) + " GiB kept free";
            ++left_;
            continue;
        }
        void* p = nullptr;
        const cudaError_t e = cudaHostAlloc(&p, size, cudaHostAllocPortable);
        if (e != cudaSuccess) {
            (void) cudaGetLastError();
            why_ = std::string("cudaHostAlloc: ") + cudaGetErrorString(e);
            ++left_;
            continue;
        }
        blocks_.push_back(p);
        bytes_ += size;
        ++layers_;
        Move mv{il, {nullptr, nullptr, nullptr}};
        for (int k = 0; k < 3; ++k) {
            const uint8_t* src = static_cast<const uint8_t*>(ts[k]->data);
            uint8_t* dst = static_cast<uint8_t*>(p) + off[k];
            mv.dst[k] = dst;
            sources.push_back({src, nullptr, n[k]});
            for (size_t o = 0; o < n[k]; o += piece) pieces.push_back({src + o, dst + o, std::min(piece, n[k] - o)});
            total += n[k];
        }
        moves.push_back(mv);
    }

    // the copies, taken by the threads in order (the disk reads ahead along each piece)
    std::atomic<size_t> next{0}, done{0};
    auto work = [&] {
        for (;;) {
            const size_t i = next.fetch_add(1);
            if (i >= pieces.size()) return;
            std::memcpy(pieces[i].dst, pieces[i].src, pieces[i].n);
            done.fetch_add(pieces[i].n);
        }
    };
    std::vector<std::thread> pool;
    for (int t = 0; t < std::max(1, threads); ++t) pool.emplace_back(work);
    auto last = std::chrono::steady_clock::now();
    while (done.load() < total) {
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
        if (progress && std::chrono::steady_clock::now() - last >= std::chrono::seconds(2)) {
            last = std::chrono::steady_clock::now();
            progress(done.load(), total);
        }
    }
    for (auto& th : pool) th.join();
    if (progress) progress(total, total);

    for (const Move& mv : moves) model.move_experts(mv.il, mv.dst[0], mv.dst[1], mv.dst[2]);
    // the mapping's pages of the moved experts are not read again: let them go (they stay in the page cache, which
    // the kernel takes back when it needs the memory)
    const size_t page = (size_t) std::max(4096L, sysconf(_SC_PAGESIZE));
    for (const Piece& s : sources) {
        const uintptr_t a = ((uintptr_t) s.src + page - 1) & ~(uintptr_t) (page - 1);
        const uintptr_t b = ((uintptr_t) s.src + s.n) & ~(uintptr_t) (page - 1);
        if (b > a) madvise((void*) a, b - a, MADV_DONTNEED);
    }
}

PinnedExperts::~PinnedExperts() {
    for (void* p : blocks_) cudaFreeHost(p);
}

}  // namespace strata::glm
