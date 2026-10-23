// tests/glm/thread_pool_test.cpp - ThreadPool (src/glm/cpu_ref.cpp): every index of every job is visited exactly
// once, in chunks of the asked size (or all at once on one thread), over many jobs in a row (the spinning workers)
// and after the workers have gone to sleep; a side task runs once and its exception comes back after the job;
// physical_cores() is a sane number.
#include "strata/glm/cpu_ref.hpp"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <stdexcept>
#include <thread>
#include <vector>
using namespace strata::glm;

int main() {
    int bad = 0;
    for (int threads : {1, 2, 7, 22}) {
        ThreadPool pool(threads);
        for (int64_t n : {0, 1, 5, 31, 32, 1000, 65537}) {
            for (int64_t grain : {1, 16, 32, 100000}) {
                std::vector<std::atomic<int>> seen((size_t) n);
                std::atomic<int> wrong_size{0};
                const bool whole = threads == 1 || n <= grain;
                pool.parallel_for(n, grain, [&](int64_t b, int64_t e) {
                    if (whole ? (b != 0 || e != n) : (b >= e || e - b > grain || (e - b < grain && e != n)))
                        ++wrong_size;
                    for (int64_t i = b; i < e; ++i) ++seen[(size_t) i];
                });
                int miss = 0;
                for (auto& s : seen) miss += s.load() != 1;
                if (miss || wrong_size) {
                    std::printf("FAIL threads %d n %lld grain %lld: %d indices not visited once, %d bad chunks\n",
                                threads, (long long) n, (long long) grain, miss, wrong_size.load());
                    ++bad;
                }
            }
        }
        // many small jobs back to back, as the engine's layers call it, then one after the workers slept
        std::atomic<int64_t> sum{0};
        for (int rep = 0; rep < 5000; ++rep)
            pool.parallel_for(64, [&](int64_t b, int64_t e) { sum += e - b; });
        std::this_thread::sleep_for(std::chrono::milliseconds(30));
        pool.parallel_for(64, [&](int64_t b, int64_t e) { sum += e - b; });
        if (sum.load() != 5001 * 64) {
            std::printf("FAIL threads %d: %lld items over 5001 jobs of 64\n", threads, (long long) sum.load());
            ++bad;
        }
    }
    // `side` runs once on the caller beside the job; an exception from it comes back after the job is done
    for (int threads : {1, 8}) {
        ThreadPool pool(threads);
        std::vector<std::atomic<int>> seen(10000);
        int sides = 0;
        pool.parallel_for((int64_t) seen.size(), 16,
                          [&](int64_t b, int64_t e) { for (int64_t i = b; i < e; ++i) ++seen[(size_t) i]; },
                          [&] { ++sides; });
        int miss = 0;
        for (auto& s : seen) miss += s.load() != 1;
        bool thrown = false;
        std::vector<std::atomic<int>> seen2(10000);
        try {
            pool.parallel_for((int64_t) seen2.size(), 16,
                              [&](int64_t b, int64_t e) { for (int64_t i = b; i < e; ++i) ++seen2[(size_t) i]; },
                              [&] { throw std::runtime_error("side"); });
        } catch (const std::runtime_error&) {
            thrown = true;
        }
        int miss2 = 0;
        for (auto& s : seen2) miss2 += s.load() != 1;
        if (sides != 1 || miss || !thrown || miss2) {
            std::printf("FAIL side, threads %d: ran %d times, %d + %d indices not visited once, rethrown %d\n", threads,
                        sides, miss, miss2, (int) thrown);
            ++bad;
        }
    }
    const int cores = ThreadPool::physical_cores();
    const int hw = (int) std::thread::hardware_concurrency();
    if (cores < 1 || (hw > 0 && cores > hw)) {
        std::printf("FAIL physical_cores %d with %d hardware threads\n", cores, hw);
        ++bad;
    }
    std::printf("%s (physical_cores %d of %d hardware threads)\n", bad ? "FAILED" : "PASS", cores, hw);
    return bad ? 1 : 0;
}
