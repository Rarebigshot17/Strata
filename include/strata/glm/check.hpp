// include/strata/glm/check.hpp - the pass rule of strata-glm's GPU checks (--check-cpu and --prefill --check-cpu):
// the GPU engine's logits against the CPU reference's on the same tokens. tests/glm/check_test.cpp tests it without
// a GPU; tools/glm53_gpu_check.sh holds glm53_compare.py to the same limits against llama.cpp.
//
// The numbers behind it come from napster-ai (2026-10-20): 36 seeds of the GPU check's small random models, the GPU
// engine against the CPU reference, and llama.cpp's CUDA backend against its own CPU backend, on the same files.
// Before a product with quantized weights the GPU rounds the activations to q8_1, as llama.cpp's CUDA backend does,
// and the tiny models carry that rounding through every layer. llama.cpp's own GPU differed from its CPU as much, so
// this is rounding, not a bug. The rule:
//
// - The worst relative diff: each row's max |cpu - gpu| over its max |cpu|, the worst row. With quantized weights
//   it was 0.0323 at worst on Q4_0 (median 0.0228), llama.cpp's own 0.0344; on Q8_0 0.0077 (llama.cpp's 0.0143).
//   The limit, 0.05, is 1.45 times llama.cpp's worst and 1.5 times Strata's. It would let a 1% error through on
//   float weights (F32, F16), whose decode differed by 0.0001 at worst, so they get their own limit. The prompt path
//   multiplies in f16 on the GPU; f16 products on the CPU (strata-glm-ref on 20 seeds' v1 turned into F16, an x86
//   CPU, 2026-10-20) moved the logits by up to 0.0010. Float weights get 0.005: 5 times that, 4 times below the
//   0.02 this rule replaces, and half of a 1% difference. BF16 counts as quantized: bf16 products moved the logits
//   by up to 0.0064 in the same test (where the indexer selected the same pools).
// - The mean KL(cpu || gpu) over the rows, from each row's softmax in double. The rounding gave at most 1.32e-6
//   (Q4_0, both engines). The limit, 1e-5, is 7.6 times that. KL grows with the square of the logit differences,
//   so that is 2.75 times them, more room than the relative limit gives (1.5 times): the rounding meets the relative
//   limit first. A bug lands far above it: the indexer's select_tail bug gave 7.6e-4 on the CPU, and pools the
//   indexer selected differently (llama.cpp's f16 cache against Strata's fp32 keys) 4e-5 to 1.5e-4.
// - A NaN or an infinity in either row fails.
//
// Top-1 agreement is printed but is not part of the rule. When every logit of a row differs by at most d, the two
// engines can pick different tokens i (CPU) and j (GPU) only when the CPU puts them less than 2d apart: b[j] >= b[i]
// means a[i] - a[j] <= e[j] - e[i] <= 2d. So the count adds nothing to the bound the relative diff sets; it counts
// near ties. The small models have many (4 to 13 rows of 40 with a top-2 gap under 0.01 on Q4_0, 20 seeds on the
// CPU), and both engines dropped to 37 of 40 there.
//
// The limits are for the GPU check's small random models. Nothing has been measured on the real model yet.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

namespace strata::glm {

/// Whether weights of ggml type `t` leave the activations they meet in f32 or f16: F32 (0) and F16 (1). Every
/// other type rounds them to 8 bits or so (q8_1 on the GPU, the type's dot-product type on the CPU, or bf16).
inline bool float_weight_type(uint32_t t) { return t == 0 || t == 1; }

struct CheckLimits {
    double rel_diff_max;    ///< the worst row's relative diff
    double mean_kl_max;     ///< the mean KL(cpu || gpu) over the rows
    const char* weights;    ///< "quantized" or "float"
};

/// The limits for a model with some weight matrix of a type float_weight_type refuses (`quantized`), or none.
inline CheckLimits check_limits(bool quantized) {
    return quantized ? CheckLimits{0.05, 1e-5, "quantized"} : CheckLimits{0.005, 1e-5, "float"};
}

inline bool all_finite(const std::vector<float>& v) {
    return std::all_of(v.begin(), v.end(), [](float x) { return std::isfinite(x); });
}

/// max |a - b| relative to max |a|; NaN when either row holds a NaN or an infinity
inline double rel_diff(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size() || !all_finite(a) || !all_finite(b)) return NAN;
    double num = 0, den = 1e-30;
    for (size_t i = 0; i < a.size(); ++i) {
        num = std::max(num, std::fabs((double) a[i] - (double) b[i]));
        den = std::max(den, std::fabs((double) a[i]));
    }
    return num / den;
}

/// KL(softmax(a) || softmax(b)), in double; NaN when either row holds a NaN or an infinity
inline double kl_div(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.empty() || a.size() != b.size() || !all_finite(a) || !all_finite(b)) return NAN;
    double ma = a[0], mb = b[0];
    for (size_t i = 0; i < a.size(); ++i) ma = std::max(ma, (double) a[i]), mb = std::max(mb, (double) b[i]);
    double sa = 0, sb = 0;
    for (size_t i = 0; i < a.size(); ++i) sa += std::exp(a[i] - ma), sb += std::exp(b[i] - mb);
    const double la = ma + std::log(sa), lb = mb + std::log(sb);   // the log of each softmax's denominator
    double kl = 0;
    for (size_t i = 0; i < a.size(); ++i) kl += std::exp(a[i] - la) * ((a[i] - la) - (b[i] - lb));
    return std::max(kl, 0.0);
}

/// The residual streams after every layer, compared position by position (--check-cpu): the first layer where the
/// GPU parts from the CPU is where a kernel is wrong. It says where to look and is not part of the rule. A NaN or an
/// infinity in a layer's streams parts it too (rel_diff gives NaN, which no comparison with a limit lets through),
/// and keeps the worst diff at NaN: an overflow is the kind of wrong a GPU kernel makes.
struct LayerCheck {
    double limit = 1e-2;    ///< a layer whose relative diff is above this has parted
    double worst = 0;       ///< over the layers compared; NaN once one held a NaN or an infinity

    /// Adds one layer's streams and returns their relative diff (NaN for a NaN or an infinity in either).
    double add(const std::vector<float>& cpu, const std::vector<float>& gpu) {
        const double d = rel_diff(cpu, gpu);
        worst = std::isnan(worst) || std::isnan(d) ? NAN : std::max(worst, d);
        return d;
    }
    /// Whether a layer whose relative diff is `d` has parted from the CPU's.
    bool parted(double d) const { return !(d <= limit); }
};

/// The CPU's and the GPU's logits, compared row by row.
struct LogitCheck {
    int rows = 0;
    int top1_same = 0;
    int bad_rows = 0;           ///< rows with a NaN or an infinity, in either
    double worst_rel = 0;       ///< over the other rows
    double kl_sum = 0;          ///< over the other rows

    /// Adds one row and returns its relative diff (NaN for a row with a NaN or an infinity).
    double add(const std::vector<float>& cpu, const std::vector<float>& gpu) {
        ++rows;
        const double r = rel_diff(cpu, gpu), kl = kl_div(cpu, gpu);
        if (std::isnan(r) || std::isnan(kl)) {
            ++bad_rows;
            return NAN;
        }
        worst_rel = std::max(worst_rel, r);
        kl_sum += kl;
        top1_same += std::max_element(cpu.begin(), cpu.end()) - cpu.begin() ==
                     std::max_element(gpu.begin(), gpu.end()) - gpu.begin();
        return r;
    }
    double mean_kl() const { return rows > bad_rows ? kl_sum / (rows - bad_rows) : NAN; }
    bool pass(const CheckLimits& l) const {
        return rows > 0 && bad_rows == 0 && worst_rel <= l.rel_diff_max && mean_kl() <= l.mean_kl_max;
    }
    /// The numbers and the limits, for the check's last line.
    std::string summary(const CheckLimits& l) const {
        char buf[320];
        int n = std::snprintf(buf, sizeof buf,
                              "worst logits relative diff %.3g (limit %g), mean KL %.3g (limit %g), top-1 agreement "
                              "%d/%d (not in the rule), %s weights",
                              worst_rel, l.rel_diff_max, mean_kl(), l.mean_kl_max, top1_same, rows, l.weights);
        if (bad_rows && n > 0 && n < (int) sizeof buf)
            std::snprintf(buf + n, sizeof buf - (size_t) n, ", %d row%s with NaN or inf", bad_rows,
                          bad_rows > 1 ? "s" : "");
        return buf;
    }
};

}  // namespace strata::glm
