// src/glm/pinned_experts.cu - the routed experts in page-locked RAM (pinned_experts.hpp).
#include "strata/glm/pinned_experts.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <cerrno>
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
        void* block;
        size_t size;
    };
    std::vector<Piece> pieces;     // the copies, in 64 MiB pieces
    std::vector<Piece> sources;    // each tensor's whole source range, for its pages to be let go after
    std::vector<Move> moves;
    const size_t piece = (size_t) 64 << 20, align = 4096, huge = (size_t) 2 << 20;
    size_t total = 0, planned = 0;
    const size_t avail0 = mem_available();
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
        // the memory is taken as it is written, so what this layer will take is counted against MemAvailable
        // before any of it is used
        size = (size + huge - 1) / huge * huge;
        if (avail0 != 0 && avail0 < planned + size + keep_free) {
            why_ = "MemAvailable " + std::to_string((avail0 - std::min(avail0, planned)) >> 30) +
                   " GiB would fall below the " + std::to_string(keep_free >> 30) + " GiB kept free";
            ++left_;
            continue;
        }
        void* p = mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (p == MAP_FAILED) {
            why_ = std::string("mmap: ") + std::strerror(errno);
            ++left_;
            continue;
        }
        madvise(p, size, MADV_HUGEPAGE);    // 2 MiB pages where the kernel allows them: fewer faults, faster locking
        planned += size;
        Move mv{il, {nullptr, nullptr, nullptr}, p, size};
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

    // then each layer's block page-locked for every GPU (cudaHostRegister, the threads one block each at a time);
    // a layer whose block cannot be is left on the mapped file
    std::vector<cudaError_t> reg(moves.size(), cudaSuccess);
    std::atomic<size_t> next_reg{0};
    auto lock = [&] {
        for (;;) {
            const size_t i = next_reg.fetch_add(1);
            if (i >= moves.size()) return;
            reg[i] = cudaHostRegister(moves[i].block, moves[i].size, cudaHostRegisterPortable);
        }
    };
    pool.clear();
    for (int t = 0; t < std::max(1, std::min(threads, 8)); ++t) pool.emplace_back(lock);
    for (auto& th : pool) th.join();
    for (size_t i = 0; i < moves.size(); ++i) {
        const Move& mv = moves[i];
        if (reg[i] != cudaSuccess) {
            (void) cudaGetLastError();
            if (why_.empty()) why_ = std::string("cudaHostRegister: ") + cudaGetErrorString(reg[i]);
            munmap(mv.block, mv.size);
            ++left_;
            continue;
        }
        blocks_.push_back({mv.block, mv.size});
        bytes_ += mv.size;
        ++layers_;
        model.move_experts(mv.il, mv.dst[0], mv.dst[1], mv.dst[2]);
    }
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
    for (const auto& [p, n] : blocks_) {
        cudaHostUnregister(p);
        munmap(p, n);
    }
}

}  // namespace strata::glm
