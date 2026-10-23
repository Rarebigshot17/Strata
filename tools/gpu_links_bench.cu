// How fast can data move between RAM and the cards, and between the cards, one at a time and all at once?  The
// answer decides how several cards can share one prompt chunk: whether each may stream its own experts from RAM
// (do the cards share one link to the CPU?) and whether a chunk's activations can go card to card directly (peer
// access) or only through RAM.  Standalone: not part of the build.
//   nvcc -O2 -o gpu_links_bench tools/gpu_links_bench.cu && ./gpu_links_bench [MiB per card, default 1024]
#include <cuda_runtime.h>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { std::printf("%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); std::exit(1); } } while (0)
using Clock = std::chrono::steady_clock;
static double sec_since(Clock::time_point t) { return std::chrono::duration<double>(Clock::now() - t).count(); }

constexpr size_t kPiece = 9u << 20;   // about one expert of the 3-bit file, the prompt path's copy size

struct Card {
    int id = 0;
    cudaStream_t s = nullptr;
    uint8_t* d = nullptr;        // device buffer, `bytes` long
    uint8_t* peer_in = nullptr;  // device buffer for peer copies into this card
};

// Copies `bytes` per listed card at once, in kPiece pieces, host->device (h2d) or device->host; returns GB/s per card
// and in total (best of 3).
static void host_copies(std::vector<Card>& cards, const std::vector<int>& which, uint8_t* host, size_t bytes, bool h2d,
                        std::vector<double>& per, double& total) {
    double best = 1e30;
    std::vector<double> best_per(which.size(), 0.0);
    for (int rep = 0; rep < 3; ++rep) {
        std::vector<cudaEvent_t> a(which.size()), b(which.size());
        for (size_t k = 0; k < which.size(); ++k) {
            Card& c = cards[(size_t) which[k]];
            CK(cudaSetDevice(c.id));
            CK(cudaEventCreate(&a[k]));
            CK(cudaEventCreate(&b[k]));
        }
        const auto t0 = Clock::now();
        for (size_t k = 0; k < which.size(); ++k) {
            Card& c = cards[(size_t) which[k]];
            CK(cudaSetDevice(c.id));
            CK(cudaEventRecord(a[k], c.s));
            uint8_t* h = host + k * bytes;   // each card its own part of the host buffer
            for (size_t off = 0; off < bytes; off += kPiece) {
                const size_t n = std::min(kPiece, bytes - off);
                if (h2d) CK(cudaMemcpyAsync(c.d + off, h + off, n, cudaMemcpyHostToDevice, c.s));
                else CK(cudaMemcpyAsync(h + off, c.d + off, n, cudaMemcpyDeviceToHost, c.s));
            }
            CK(cudaEventRecord(b[k], c.s));
        }
        for (size_t k = 0; k < which.size(); ++k) {
            CK(cudaSetDevice(cards[(size_t) which[k]].id));
            CK(cudaStreamSynchronize(cards[(size_t) which[k]].s));
        }
        const double t = sec_since(t0);
        if (t < best) {
            best = t;
            for (size_t k = 0; k < which.size(); ++k) {
                float ms = 0;
                CK(cudaEventElapsedTime(&ms, a[k], b[k]));
                best_per[k] = (double) bytes / (ms / 1e3) / 1e9;
            }
        }
        for (size_t k = 0; k < which.size(); ++k) {
            CK(cudaEventDestroy(a[k]));
            CK(cudaEventDestroy(b[k]));
        }
    }
    per = best_per;
    total = (double) bytes * (double) which.size() / best / 1e9;
}

static std::string names(const std::vector<int>& which) {
    std::string s;
    for (int i : which) s += (s.empty() ? "" : "+") + std::to_string(i);
    return s;
}

int main(int argc, char** argv) {
    const size_t mib = argc > 1 ? std::strtoull(argv[1], nullptr, 10) : 1024;
    const size_t bytes = (std::max<size_t>(mib, 16) << 20) / kPiece * kPiece;
    int n = 0;
    CK(cudaGetDeviceCount(&n));
    if (n < 1) { std::printf("no CUDA devices\n"); return 1; }
    std::vector<Card> cards((size_t) n);
    for (int i = 0; i < n; ++i) {
        cudaDeviceProp p;
        CK(cudaGetDeviceProperties(&p, i));
        std::printf("card %d: %s, PCI %04x:%02x:%02x, copy engines %d\n", i, p.name, p.pciDomainID, p.pciBusID,
                    p.pciDeviceID, p.asyncEngineCount);
        cards[(size_t) i].id = i;
        CK(cudaSetDevice(i));
        CK(cudaStreamCreateWithFlags(&cards[(size_t) i].s, cudaStreamNonBlocking));
        CK(cudaMalloc(&cards[(size_t) i].d, bytes));
        CK(cudaMalloc(&cards[(size_t) i].peer_in, bytes));
        CK(cudaMemset(cards[(size_t) i].d, 1, bytes));
    }
    uint8_t* host = nullptr;
    CK(cudaHostAlloc(&host, bytes * (size_t) n, cudaHostAllocPortable));
    for (size_t i = 0; i < bytes * (size_t) n; i += 4096) host[i] = (uint8_t) i;
    std::printf("%zu MiB per card per test, in pieces of %zu MiB, best of 3\n\n", bytes >> 20, kPiece >> 20);

    // 1. RAM -> card and card -> RAM: each card alone, then every pair, then all at once
    std::vector<std::vector<int>> sets;
    for (int i = 0; i < n; ++i) sets.push_back({i});
    for (int i = 0; i < n; ++i)
        for (int j = i + 1; j < n; ++j) sets.push_back({i, j});
    if (n > 2) {
        std::vector<int> all;
        for (int i = 0; i < n; ++i) all.push_back(i);
        sets.push_back(all);
    }
    for (int dir = 0; dir < 2; ++dir) {
        std::printf("%s (GB/s)\n", dir == 0 ? "RAM -> card" : "card -> RAM");
        for (const auto& w : sets) {
            std::vector<double> per;
            double total = 0;
            host_copies(cards, w, host, bytes, dir == 0, per, total);
            std::printf("  cards %-6s total %6.2f   each:", names(w).c_str(), total);
            for (size_t k = 0; k < w.size(); ++k) std::printf(" %d: %.2f", w[k], per[k]);
            std::printf("\n");
        }
    }

    // 2. card -> card: can each pair access the other directly, and how fast is a copy with and without that
    std::printf("\npeer access (row can read/write column directly):\n      ");
    for (int j = 0; j < n; ++j) std::printf(" %d", j);
    std::printf("\n");
    std::vector<std::vector<int>> can((size_t) n, std::vector<int>((size_t) n, 0));
    for (int i = 0; i < n; ++i) {
        std::printf("  %d:  ", i);
        for (int j = 0; j < n; ++j) {
            if (i != j) CK(cudaDeviceCanAccessPeer(&can[(size_t) i][(size_t) j], i, j));
            std::printf(" %s", i == j ? "-" : can[(size_t) i][(size_t) j] ? "Y" : "n");
        }
        std::printf("\n");
    }
    for (int mode = 0; mode < 2; ++mode) {
        if (mode == 1) {
            bool any = false;
            for (int i = 0; i < n; ++i)
                for (int j = 0; j < n; ++j)
                    if (i != j && can[(size_t) i][(size_t) j]) {
                        CK(cudaSetDevice(i));
                        cudaError_t e = cudaDeviceEnablePeerAccess(j, 0);
                        if (e != cudaSuccess && e != cudaErrorPeerAccessAlreadyEnabled)
                            std::printf("  enabling %d->%d: %s\n", i, j, cudaGetErrorString(e));
                        (void) cudaGetLastError();
                        any = true;
                    }
            if (!any) {
                std::printf("no pair has peer access: every card-to-card copy goes through RAM\n");
                break;
            }
        }
        std::printf("card -> card (GB/s), %s:\n", mode == 0 ? "peer access not enabled" : "peer access enabled");
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j) {
                if (i == j) continue;
                if (mode == 1 && !can[(size_t) j][(size_t) i] && !can[(size_t) i][(size_t) j]) continue;
                double best = 1e30;
                for (int rep = 0; rep < 3; ++rep) {
                    CK(cudaSetDevice(i));
                    CK(cudaDeviceSynchronize());
                    const auto t0 = Clock::now();
                    CK(cudaMemcpyPeerAsync(cards[(size_t) j].peer_in, j, cards[(size_t) i].d, i, bytes, cards[(size_t) i].s));
                    CK(cudaStreamSynchronize(cards[(size_t) i].s));
                    best = std::min(best, sec_since(t0));
                }
                std::printf("  %d -> %d: %6.2f\n", i, j, (double) bytes / best / 1e9);
            }
    }

    // 3. the mix a shared chunk would make: card 0 streams from RAM while card 0 sends to the others
    if (n > 1) {
        std::printf("\nRAM -> card 0 while card 0 -> card 1 runs (GB/s, peer access as above):\n");
        cudaStream_t s2;
        CK(cudaSetDevice(0));
        CK(cudaStreamCreateWithFlags(&s2, cudaStreamNonBlocking));
        double best = 1e30;
        for (int rep = 0; rep < 3; ++rep) {
            CK(cudaDeviceSynchronize());
            const auto t0 = Clock::now();
            for (size_t off = 0; off < bytes; off += kPiece)
                CK(cudaMemcpyAsync(cards[0].d + off, host + off, std::min(kPiece, bytes - off), cudaMemcpyHostToDevice,
                                   cards[0].s));
            CK(cudaMemcpyPeerAsync(cards[1].peer_in, 1, cards[0].d, 0, bytes, s2));
            CK(cudaStreamSynchronize(cards[0].s));
            CK(cudaStreamSynchronize(s2));
            best = std::min(best, sec_since(t0));
        }
        std::printf("  both done in %.3f s: %.2f GB/s moved in total\n", best, 2.0 * (double) bytes / best / 1e9);
        CK(cudaStreamDestroy(s2));
    }
    CK(cudaFreeHost(host));
    return 0;
}
