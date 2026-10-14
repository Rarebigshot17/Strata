// src/glm/cpu_experts.cpp - the routed experts on the CPU (cpu_experts.hpp).
#include "strata/glm/cpu_experts.hpp"

#include "ggml.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>

namespace strata::glm {

namespace {

const ggml_type_traits_cpu* dot_traits(const Tensor& t) {
    const auto* tr = ggml_get_type_traits_cpu((ggml_type) t.type);
    if (!tr || !tr->vec_dot) throw std::runtime_error("ggml-cpu has no dot product for " + t.name);
    return tr;
}

/// x quantized to `t`'s vec_dot_type into `buf`.
void quant_act(const Tensor& t, const float* x, int64_t n, std::vector<uint8_t>& buf) {
    const ggml_type at = dot_traits(t)->vec_dot_type;
    buf.resize(ggml_row_size(at, n));
    if (at == GGML_TYPE_F32) std::memcpy(buf.data(), x, (size_t) n * sizeof(float));
    else ggml_get_type_traits_cpu(at)->from_float(x, buf.data(), n);
}

}  // namespace

CpuExperts::CpuExperts(const Model& model, ThreadPool& pool) : m_(model), pool_(pool) { ggml_cpu_init(); }

void CpuExperts::run(int64_t il, const float* x, const int32_t* ids, const float* weights, int k, float* out) {
    const Config& c = m_.config();
    const LayerWeights& w = m_.layer(il);
    const int64_t E = c.n_embd, F = c.n_ff_exp[(size_t) il];
    const float limit = c.clamp_exp[(size_t) il];
    const auto* tg = dot_traits(w.gate_exps);
    const auto* tu = dot_traits(w.up_exps);
    const auto* td = dot_traits(w.down_exps);
    quant_act(w.gate_exps, x, E, xq_g_);
    quant_act(w.up_exps, x, E, xq_u_);
    g_.resize((size_t) (k * F));
    u_.resize((size_t) (k * F));
    d_.resize((size_t) (k * E));

    // gate and up rows of every selected expert: 2 * k * F dot products
    pool_.parallel_for(2 * k * F, [&](int64_t b, int64_t e) {
        for (int64_t r = b; r < e; ++r) {
            const bool up = r >= k * F;
            const int64_t rr = up ? r - k * F : r;
            const int64_t j = rr / F, row = rr % F;
            const Tensor& t = up ? w.up_exps : w.gate_exps;
            const uint8_t* wr = t.slice2(ids[j]).row(row);
            float sum = 0.0f;
            (up ? tu : tg)->vec_dot((int) E, &sum, 0, wr, 0, (up ? xq_u_ : xq_g_).data(), 0, 1);
            (up ? u_ : g_)[(size_t) rr] = sum;
        }
    });
    // the clamped SwiGLU, then each expert's activation in the down matrix's format
    for (int64_t i = 0; i < k * F; ++i) {
        float gi = g_[(size_t) i], ui = u_[(size_t) i];
        if (limit > 1e-6f) {
            gi = std::min(gi, limit);
            ui = std::clamp(ui, -limit, limit);
        }
        g_[(size_t) i] = gi / (1.0f + std::exp(-gi)) * ui;
    }
    const ggml_type hat = td->vec_dot_type;
    const size_t hrow = ggml_row_size(hat, F);
    hq_.resize(hrow * (size_t) k);
    for (int j = 0; j < k; ++j) {
        if (hat == GGML_TYPE_F32) std::memcpy(hq_.data() + hrow * j, g_.data() + j * F, (size_t) F * sizeof(float));
        else ggml_get_type_traits_cpu(hat)->from_float(g_.data() + j * F, hq_.data() + hrow * j, F);
    }
    pool_.parallel_for(k * E, [&](int64_t b, int64_t e) {
        for (int64_t r = b; r < e; ++r) {
            const int64_t j = r / E, row = r % E;
            float sum = 0.0f;
            td->vec_dot((int) F, &sum, 0, w.down_exps.slice2(ids[j]).row(row), 0, hq_.data() + hrow * j, 0, 1);
            d_[(size_t) r] = sum;
        }
    });
    // the weighted sum, experts in routing order (cpu_ref's order)
    std::fill(out, out + E, 0.0f);
    for (int j = 0; j < k; ++j)
        for (int64_t i = 0; i < E; ++i) out[i] += weights[j] * d_[(size_t) (j * E + i)];
}

}  // namespace strata::glm
