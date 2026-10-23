// src/glm/cpu_experts.cpp - the routed experts on the CPU (cpu_experts.hpp).
#include "strata/glm/cpu_experts.hpp"

#include "strata/glm/cpu_dot.hpp"

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
    // cpu_dot.hpp's kernel where there is one (bitwise ggml's), else ggml-cpu's
    const VecDotFn dot_g = fast_vec_dot((int) w.gate_exps.type) ? fast_vec_dot((int) w.gate_exps.type) : tg->vec_dot;
    const VecDotFn dot_u = fast_vec_dot((int) w.up_exps.type) ? fast_vec_dot((int) w.up_exps.type) : tu->vec_dot;
    const VecDotFn dot_d = fast_vec_dot((int) w.down_exps.type) ? fast_vec_dot((int) w.down_exps.type) : td->vec_dot;
    quant_act(w.gate_exps, x, E, xq_g_);
    quant_act(w.up_exps, x, E, xq_u_);
    g_.resize((size_t) (k * F));
    u_.resize((size_t) (k * F));
    gate_.resize((size_t) k);
    up_.resize((size_t) k);
    down_.resize((size_t) k);
    for (int j = 0; j < k; ++j) {
        gate_[(size_t) j] = w.gate_exps.slice2(ids[j]);
        up_[(size_t) j] = w.up_exps.slice2(ids[j]);
        down_[(size_t) j] = w.down_exps.slice2(ids[j]);
    }

    // gate and up rows of every selected expert: 2 * k * F dot products, 32 rows a chunk
    pool_.parallel_for(2 * k * F, 32, [&](int64_t b, int64_t e) {
        for (int64_t r = b; r < e; ++r) {
            const bool up = r >= k * F;
            const int64_t rr = up ? r - k * F : r;
            const int64_t j = rr / F, row = rr % F;
            const uint8_t* wr = (up ? up_ : gate_)[(size_t) j].row(row);
            float sum = 0.0f;
            (up ? dot_u : dot_g)((int) E, &sum, 0, wr, 0, (up ? xq_u_ : xq_g_).data(), 0, 1);
            (up ? u_ : g_)[(size_t) rr] = sum;
        }
    });
    // the clamped SwiGLU, then each expert's activation in the down matrix's format, in pieces of 256 values (a
    // whole number of blocks of every activation format, each block quantized on its own as in a whole-row call)
    const ggml_type hat = td->vec_dot_type;
    const size_t hrow = ggml_row_size(hat, F);
    const int64_t piece = F % 256 == 0 && 256 % ggml_blck_size(hat) == 0 ? 256 : F;
    const int64_t pieces = F / piece;
    hq_.resize(hrow * (size_t) k);
    pool_.parallel_for(k * pieces, 1, [&](int64_t b, int64_t e) {
        for (int64_t q = b; q < e; ++q) {
            const int64_t j = q / pieces, i0 = (q % pieces) * piece;
            float* h = g_.data() + j * F + i0;
            const float* uu = u_.data() + j * F + i0;
            for (int64_t i = 0; i < piece; ++i) {
                float gi = h[i], ui = uu[i];
                if (limit > 1e-6f) {
                    gi = std::min(gi, limit);
                    ui = std::clamp(ui, -limit, limit);
                }
                h[i] = gi / (1.0f + std::exp(-gi)) * ui;
            }
            uint8_t* dst = hq_.data() + hrow * j + ggml_row_size(hat, i0);
            if (hat == GGML_TYPE_F32) std::memcpy(dst, h, (size_t) piece * sizeof(float));
            else ggml_get_type_traits_cpu(hat)->from_float(h, dst, piece);
        }
    });
    // each output row: its k down rows, summed in routing order (cpu_ref's order), 16 rows a chunk
    pool_.parallel_for(E, 16, [&](int64_t b, int64_t e) {
        for (int64_t row = b; row < e; ++row) {
            float acc = 0.0f;
            for (int j = 0; j < k; ++j) {
                float sum = 0.0f;
                dot_d((int) F, &sum, 0, down_[(size_t) j].row(row), 0, hq_.data() + hrow * j, 0, 1);
                acc += weights[j] * sum;
            }
            out[row] = acc;
        }
    });
}

}  // namespace strata::glm
