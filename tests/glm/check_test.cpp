// tests/glm/check_test.cpp - the pass rule of strata-glm's GPU checks (check.hpp) on hand-made logit rows, without a
// GPU: the spread measured on napster-ai's Q4_0 models passes, a 1% difference fails on float weights, a KL far above
// the rounding fails while the relative diff is within its limit, and a NaN or an infinity fails. Also the layers'
// residual streams (LayerCheck): one with a NaN or an infinity parts from the CPU's.
#include "strata/glm/check.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

using namespace strata::glm;

namespace {

int bad = 0;

void expect(bool ok, const char* what) {
    std::printf("  %s: %s\n", ok ? "ok  " : "FAIL", what);
    bad += !ok;
}

/// uniform in [0, 1) from mt19937's raw output, the same on every platform
double uni(std::mt19937& g) { return (double) (g() >> 8) * (1.0 / 16777216.0); }

/// rows x n logits shaped like the GPU check's small models: mean 0, the largest |logit| about 0.3
std::vector<std::vector<float>> cpu_rows(int rows, int n, unsigned seed) {
    std::mt19937 g(seed);
    std::vector<std::vector<float>> out((size_t) rows, std::vector<float>((size_t) n));
    for (auto& r : out)
        for (auto& v : r) v = (float) (0.1 * (uni(g) + uni(g) + uni(g) + uni(g) - 2.0) * std::sqrt(3.0));
    return out;
}

float row_max(const std::vector<float>& r) {
    float m = 0;
    for (float v : r) m = std::max(m, std::fabs(v));
    return m;
}

int argmax(const std::vector<float>& r) { return (int) (std::max_element(r.begin(), r.end()) - r.begin()); }

int arg_second(const std::vector<float>& r) {
    const int top = argmax(r);
    int j = top == 0 ? 1 : 0;
    for (int i = 0; i < (int) r.size(); ++i)
        if (i != top && r[i] > r[j]) j = i;
    return j;
}

int argmin(const std::vector<float>& r) { return (int) (std::min_element(r.begin(), r.end()) - r.begin()); }

LogitCheck compare(const std::vector<std::vector<float>>& cpu, const std::vector<std::vector<float>>& gpu) {
    LogitCheck c;
    for (size_t i = 0; i < cpu.size(); ++i) c.add(cpu[i], gpu[i]);
    return c;
}

}  // namespace

int main() {
    const CheckLimits quant = check_limits(true), flt = check_limits(false);

    std::printf("the weight types\n");
    expect(float_weight_type(0) && float_weight_type(1), "F32 and F16 are float");
    expect(!float_weight_type(2) && !float_weight_type(8) && !float_weight_type(14) && !float_weight_type(30),
           "Q4_0, Q8_0, Q6_K and BF16 are not");

    std::printf("the Q4_0 spread measured on napster-ai (worst row 0.0323, mean KL about 1.3e-6, top-1 37/40)\n");
    {
        auto cpu = cpu_rows(40, 128, 1);
        auto gpu = cpu;
        std::mt19937 g(2);
        // rounding on every logit, about 0.0016 RMS (uniform within +-0.0028)
        for (auto& r : gpu)
            for (auto& v : r) v += (float) (0.0028 * (2 * uni(g) - 1));
        // one row whose least likely token is off by 3.23% of the row's largest logit
        gpu[11][(size_t) argmin(cpu[11])] = cpu[11][(size_t) argmin(cpu[11])] + 0.0323f * row_max(cpu[11]);
        // three rows whose two most likely tokens swap places: 4% of the row's largest apart on the CPU (about 0.012,
        // more than glm53_compare.py's tie margin), each moved by 2.5% of it
        for (int r : {3, 17, 29}) {
            const float m = row_max(cpu[(size_t) r]);
            const int i = argmax(cpu[(size_t) r]), j = arg_second(cpu[(size_t) r]);
            cpu[(size_t) r][(size_t) j] = cpu[(size_t) r][(size_t) i] - 0.04f * m;
            gpu[(size_t) r][(size_t) i] = cpu[(size_t) r][(size_t) i] - 0.025f * m;
            gpu[(size_t) r][(size_t) j] = cpu[(size_t) r][(size_t) j] + 0.025f * m;
        }
        const LogitCheck c = compare(cpu, gpu);
        std::printf("  %s\n", c.summary(quant).c_str());
        expect(std::fabs(c.worst_rel - 0.0323) < 1e-4, "the worst row is the measured 0.0323");
        expect(c.mean_kl() > 5e-7 && c.mean_kl() < 2e-6, "the mean KL is about the measured 1.3e-6");
        expect(c.top1_same <= 37, "the top-1 agreement is 37/40 or less");
        expect(c.pass(quant), "passes with the quantized weights' limits");
        expect(!c.pass(flt), "fails with the float weights' limits");
        expect(c.summary(quant).find("(limit 0.05)") != std::string::npos, "the summary gives the limit");
    }

    std::printf("float weights\n");
    {
        const auto cpu = cpu_rows(40, 128, 3);
        auto gpu = cpu;
        gpu[5][7] += 0.01f * row_max(cpu[5]);
        const LogitCheck c = compare(cpu, gpu);
        std::printf("  %s\n", c.summary(flt).c_str());
        expect(!c.pass(flt), "a 1% difference in one row fails");
        expect(c.pass(quant), "(the quantized limits let it through: the limit depends on the weights)");
        // the decode path's measured float spread (0.0001) and the prompt path's f16 products (0.001 on the CPU)
        for (double rel : {1e-4, 1e-3}) {
            auto near = cpu;
            std::mt19937 g(4);
            for (size_t r = 0; r < near.size(); ++r) {
                const float m = row_max(cpu[r]);
                for (auto& v : near[r]) v += (float) (rel * m * (2 * uni(g) - 1));
            }
            char what[96];
            std::snprintf(what, sizeof what, "a spread of %g passes", rel);
            expect(compare(cpu, near).pass(flt), what);
        }
    }

    std::printf("a KL far above the rounding, every logit within the relative limit\n");
    {
        const auto cpu = cpu_rows(40, 128, 5);
        auto gpu = cpu;
        // each logit 4% of the row's largest up or down: what a wrong scale or a dropped term looks like
        for (size_t r = 0; r < gpu.size(); ++r) {
            const float m = row_max(cpu[r]);
            for (size_t i = 0; i < gpu[r].size(); ++i) gpu[r][i] += (i % 2 ? 0.04f : -0.04f) * m;
        }
        const LogitCheck c = compare(cpu, gpu);
        std::printf("  %s\n", c.summary(quant).c_str());
        expect(c.worst_rel <= quant.rel_diff_max, "the relative diff is within its limit");
        expect(c.mean_kl() > 2e-5, "the mean KL is far above the rounding's 1.3e-6");
        expect(!c.pass(quant), "fails");
    }

    std::printf("NaN and infinity\n");
    for (int k = 0; k < 4; ++k) {
        auto cpu = cpu_rows(40, 128, 6);
        auto gpu = cpu;
        const char* what[] = {"one NaN in a GPU row fails", "one NaN in a CPU row fails",
                              "an infinity in a GPU row fails", "a GPU row of NaN fails"};
        if (k == 0) gpu[9][40] = NAN;
        if (k == 1) cpu[9][40] = NAN;
        if (k == 2) gpu[9][40] = INFINITY;
        if (k == 3) std::fill(gpu[9].begin(), gpu[9].end(), NAN);
        const LogitCheck c = compare(cpu, gpu);
        std::printf("  %s\n", c.summary(quant).c_str());
        expect(!c.pass(quant) && !c.pass(flt), what[k]);
        expect(c.summary(quant).find("NaN or inf") != std::string::npos, "the summary says so");
    }
    expect(std::isnan(rel_diff({1.0f, 2.0f}, {1.0f, NAN})), "rel_diff of a row with a NaN is NaN");

    std::printf("no rows\n");
    expect(!LogitCheck().pass(quant), "nothing compared does not pass");

    // --check-cpu prints the first layer of each position whose residual streams part: where a kernel went wrong,
    // and an overflow or a NaN is the kind of wrong a GPU kernel makes
    std::printf("the layers' residual streams\n");
    {
        const auto cpu = cpu_rows(1, 512, 7)[0];
        auto near = cpu, far = cpu;
        near[3] += 0.005f * row_max(cpu);
        far[3] += 0.02f * row_max(cpu);
        LayerCheck lc;
        expect(!lc.parted(lc.add(cpu, near)), "a layer within 1% has not parted");
        expect(lc.parted(lc.add(cpu, far)), "a layer 2% off has parted");
        for (int k = 0; k < 3; ++k) {
            auto bad_row = cpu;
            bad_row[100] = k == 0 ? INFINITY : k == 1 ? -INFINITY : NAN;
            const char* what[] = {"an infinity", "a minus infinity", "a NaN"};
            LayerCheck c;
            c.add(cpu, near);
            char msg[96];
            std::snprintf(msg, sizeof msg, "a layer with %s on the GPU has parted", what[k]);
            expect(c.parted(c.add(cpu, bad_row)), msg);
            c.add(cpu, near);   // a later layer, or position, that is close again
            std::snprintf(msg, sizeof msg, "the worst layer diff still shows %s", what[k]);
            expect(!(c.worst <= c.limit), msg);
        }
        // the limit follows the weights, as the logits' does: on napster-ai q4_0's first layer differed from the
        // CPU's by 1.1% to 1.7% on the GPU in runs whose logits were within the rounding
        const LayerCheck q = layer_check(quant), f = layer_check(flt);
        auto q4 = cpu, q_far = cpu, f_off = cpu;
        q4[3] += 0.0172f * row_max(cpu);
        q_far[3] += 0.06f * row_max(cpu);
        f_off[3] += 0.01f * row_max(cpu);
        expect(!q.parted(rel_diff(cpu, q4)), "q4_0's 1.7% has not parted with quantized weights");
        expect(q.parted(rel_diff(cpu, q_far)), "6% has parted with quantized weights");
        expect(f.parted(rel_diff(cpu, f_off)), "1% has parted with float weights");
        expect(!f.parted(rel_diff(cpu, cpu)), "the same streams have not parted");
    }

    std::printf(bad ? "FAILED\n" : "all ok\n");
    return bad != 0;
}
