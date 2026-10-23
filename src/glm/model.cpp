// src/glm/model.cpp - reads a GLM-5.3-Flash GGUF's configuration and resolves its tensors (model.hpp).
//
// Key names and defaults follow llama.cpp's `llama_model_glm5_next::load_arch_hparams` and the generic hparams
// loader: an array key gives one value per layer, a scalar key gives the same value for every layer.
#include "strata/glm/model.hpp"

#include "ggml.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <sstream>
#include <stdexcept>
#include <thread>

#include <sys/mman.h>
#include <unistd.h>

namespace strata::glm {

size_t Tensor::row_bytes() const { return ggml_row_size((ggml_type) type, ne[0]); }

Tensor Tensor::slice2(int64_t i2) const {
    Tensor t = *this;
    t.data = static_cast<const uint8_t*>(data) + (size_t) i2 * (size_t) ne[1] * row_bytes();
    t.ne[2] = 1;
    t.ne[3] = 1;
    return t;
}

namespace {

std::string shape_str(const int64_t* ne, int n) {
    std::ostringstream s;
    s << '[';
    for (int i = 0; i < n; ++i) s << (i ? ", " : "") << ne[i];
    s << ']';
    return s.str();
}

/// A metadata value as a number, from a scalar or from element `i` of an array.
bool meta_num(const strata::MetaValue& v, size_t i, double& out) {
    if (v.type == strata::MetaType::ARRAY) {
        if (i >= v.items.size()) return false;
        out = v.items[i].num();
        return v.items[i].is_num();
    }
    if (!v.is_num()) return false;
    out = v.num();
    return true;
}

}  // namespace

Model::Model(const std::string& first_shard) {
    gguf_ = std::make_unique<strata::GgufModel>(strata::GgufModel::open(first_shard));
    read_config();
    resolve();
}

void Model::read_config() {
    const strata::GgufFile& m = gguf_->meta();
    const strata::MetaValue* arch = m.get("general.architecture");
    if (!arch) throw std::runtime_error("GGUF has no general.architecture");
    cfg_.arch = arch->s;
    if (cfg_.arch != "glm5-next" && cfg_.arch != "glm5next")
        throw std::runtime_error("architecture is '" + cfg_.arch + "'; this engine runs GLM-5.3-Flash (glm5-next)");
    const std::string p = cfg_.arch + ".";

    auto find = [&](const char* key) { return m.get(p + key); };
    auto need = [&](const char* key) -> const strata::MetaValue& {
        const strata::MetaValue* v = find(key);
        if (!v) throw std::runtime_error("GGUF is missing " + p + key);
        return *v;
    };
    auto num = [&](const char* key, size_t i = 0) {
        double d = 0;
        if (!meta_num(need(key), i, d)) throw std::runtime_error("GGUF key " + p + key + " is not a number");
        return d;
    };
    auto num_or = [&](const char* key, double def, size_t i = 0) {
        const strata::MetaValue* v = find(key);
        double d = def;
        if (v && !meta_num(*v, i, d)) throw std::runtime_error("GGUF key " + p + key + " is not a number");
        return d;
    };

    cfg_.n_layer_all = (int64_t) num("block_count");
    cfg_.n_layer_nextn = (int64_t) num_or("nextn_predict_layers", 0);
    cfg_.n_layer = cfg_.n_layer_all - cfg_.n_layer_nextn;
    if (cfg_.n_layer <= 0) throw std::runtime_error("block_count leaves no trunk layers");
    if (cfg_.n_layer_all > 64)
        throw std::runtime_error("more than 64 layers: the GGUF reader keeps only the first 64 array elements");
    cfg_.n_embd = (int64_t) num("embedding_length");
    cfg_.n_ctx_train = (int64_t) num_or("context_length", 0);
    cfg_.rms_eps = (float) num("attention.layer_norm_rms_epsilon");
    cfg_.norm_eps = (float) num_or("attention.layer_norm_epsilon", cfg_.norm_eps);

    const size_t L = (size_t) cfg_.n_layer_all;
    cfg_.n_head.resize(L);
    cfg_.n_head_kv.resize(L);
    cfg_.n_ff_exp.resize(L);
    cfg_.clamp_exp.assign(L, 0.0f);
    cfg_.clamp_shexp.assign(L, 0.0f);
    cfg_.indexer_full.assign(L, 1);
    const bool has_clamp_shexp = find("swiglu_clamp_shexp") != nullptr;
    for (size_t i = 0; i < L; ++i) {
        cfg_.n_head[i] = (int64_t) num("attention.head_count", i);
        cfg_.n_head_kv[i] = (int64_t) num("attention.head_count_kv", i);
        cfg_.n_ff_exp[i] = (int64_t) num_or("expert_feed_forward_length", 0, i);
        cfg_.clamp_exp[i] = (float) num_or("swiglu_clamp_exp", 0, i);
        cfg_.clamp_shexp[i] = has_clamp_shexp ? (float) num_or("swiglu_clamp_shexp", 0, i) : cfg_.clamp_exp[i];
        if (i < (size_t) cfg_.n_layer && find("attention.indexer.types"))
            cfg_.indexer_full[i] = num("attention.indexer.types", i) != 0 ? 1 : 0;
    }

    cfg_.n_ff = (int64_t) num_or("feed_forward_length", 0);
    cfg_.n_layer_dense_lead = (int64_t) num_or("leading_dense_block_count", 0);
    cfg_.n_expert = (int64_t) num("expert_count");
    cfg_.n_expert_used = (int64_t) num("expert_used_count");
    cfg_.n_expert_shared = (int64_t) num("expert_shared_count");
    cfg_.expert_weights_scale = (float) num_or("expert_weights_scale", 0);
    cfg_.expert_weights_norm = num_or("expert_weights_norm", 0) != 0;
    cfg_.expert_gating_func = (int) num_or("expert_gating_func", 0);
    if (cfg_.expert_gating_func == 0) cfg_.expert_gating_func = 2;   // llama.cpp: NONE means sigmoid here
    if (cfg_.expert_gating_func != 1 && cfg_.expert_gating_func != 2)
        throw std::runtime_error("expert_gating_func " + std::to_string(cfg_.expert_gating_func) +
                                 " is not supported (1 softmax, 2 sigmoid)");

    cfg_.kda_head_dim = (int64_t) num("kda.head_dim");
    cfg_.ssm_d_conv = (int64_t) num("ssm.conv_kernel");
    if (find("kda.gate_lower_bound")) {
        cfg_.kda_gate_lower_bound = (float) num("kda.gate_lower_bound");
        cfg_.has_gate_lower_bound = std::isfinite(cfg_.kda_gate_lower_bound);
    }

    cfg_.q_lora_rank = (int64_t) num("attention.q_lora_rank");
    cfg_.kv_lora_rank = (int64_t) num("attention.kv_lora_rank");
    cfg_.head_k_mla = (int64_t) num("attention.key_length_mla");
    cfg_.head_v_mla = (int64_t) num("attention.value_length_mla");
    cfg_.n_rot = (int64_t) num_or("rope.dimension_count", 0);
    if (cfg_.n_rot != 0) throw std::runtime_error("rope.dimension_count is not 0: GLM-5.3-Flash's MLA has no RoPE");

    cfg_.indexer_n_head = (int64_t) num("attention.indexer.head_count");
    cfg_.indexer_head_size = (int64_t) num("attention.indexer.key_length");
    cfg_.indexer_top_k = (int64_t) num("attention.indexer.top_k");
    cfg_.indexer_kpool = (int64_t) num("attention.indexer.kpool");
    // absent (as in the released GGUF) means select the tail, llama.cpp's default
    cfg_.indexer_select_tail = num_or("attention.indexer.kpool_select_tail", 1) != 0;
    if (cfg_.indexer_kpool <= 1 || cfg_.indexer_top_k % cfg_.indexer_kpool != 0)
        throw std::runtime_error("indexer kpool must be > 1 and divide top_k");

    cfg_.hc = (int64_t) num("hyper_connection.count");
    cfg_.hc_sinkhorn_iters = (int64_t) num("hyper_connection.sinkhorn_iterations");
    cfg_.hc_eps = (float) num("hyper_connection.epsilon");
    if (cfg_.hc < 1) throw std::runtime_error("hyper_connection.count must be at least 1");

    // the first layer must run its own indexer; a shared layer reuses the previous full one
    for (int64_t il = 0, have = 0; il < cfg_.n_layer; ++il) {
        if (cfg_.is_kda(il)) continue;
        if (cfg_.indexer_full[(size_t) il]) have = 1;
        else if (!have) throw std::runtime_error("layer " + std::to_string(il) + " shares an indexer that no earlier "
                                                 "layer computes");
    }
}

Tensor Model::get(const std::string& name, std::vector<int64_t> shape, bool required) const {
    size_t shard = 0;
    const strata::TensorInfo* ti = gguf_->find(name, &shard);
    if (!ti) {
        if (required) throw std::runtime_error("GGUF is missing tensor " + name);
        return {};
    }
    if (!gguf_->in_bounds(*ti, shard))
        throw std::runtime_error("tensor " + name + " lies outside its file or has an unknown type");
    Tensor t;
    t.name = name;
    t.type = ti->type;
    t.data = gguf_->shard(shard).tensor_data(*ti);
    for (size_t i = 0; i < ti->shape.size() && i < 4; ++i) t.ne[i] = (int64_t) ti->shape[i];
    // compare without trailing 1s: {d_conv, 1, d_inner} and {d_conv, d_inner} are both accepted for the convs
    std::vector<int64_t> have(t.ne, t.ne + 4), want = shape;
    auto squeeze = [](std::vector<int64_t>& v) {
        std::vector<int64_t> r;
        for (int64_t x : v) if (x != 1) r.push_back(x);
        v = r;
    };
    squeeze(have);
    squeeze(want);
    if (have != want)
        throw std::runtime_error("tensor " + name + " has shape " + shape_str(t.ne, (int) ti->shape.size()) +
                                 ", expected " + shape_str(shape.data(), (int) shape.size()));
    return t;
}

void Model::resolve() {
    // the vocabulary is the embedding's second dimension
    {
        const strata::TensorInfo* ti = gguf_->find("token_embd.weight");
        if (!ti || ti->shape.size() < 2) throw std::runtime_error("GGUF is missing tensor token_embd.weight");
        cfg_.n_vocab = (int64_t) ti->shape[1];
    }
    const Config& c = cfg_;
    const int64_t E = c.n_embd, V = c.n_vocab;
    tok_embd_ = get("token_embd.weight", {E, V});
    output_norm_ = get("output_norm.weight", {E});
    output_ = get("output.weight", {E, V});

    const int64_t hc = c.hc, mix = c.hc_mix_dim();
    layers_.resize((size_t) c.n_layer);
    for (int64_t il = 0; il < c.n_layer; ++il) {
        LayerWeights& w = layers_[(size_t) il];
        auto n = [&](const char* suffix) { return "blk." + std::to_string(il) + "." + suffix; };
        const int64_t H = c.n_head[(size_t) il];

        w.attn_norm = get(n("attn_norm.weight"), {E});
        w.ffn_norm = get(n("ffn_norm.weight"), {E});
        w.hc_attn_fn = get(n("hc_attn_fn.weight"), {hc * E, mix});
        w.hc_attn_base = get(n("hc_attn_base.weight"), {mix});
        w.hc_attn_scale = get(n("hc_attn_scale.weight"), {3});
        w.hc_ffn_fn = get(n("hc_ffn_fn.weight"), {hc * E, mix});
        w.hc_ffn_base = get(n("hc_ffn_base.weight"), {mix});
        w.hc_ffn_scale = get(n("hc_ffn_scale.weight"), {3});

        if (c.is_kda(il)) {
            const int64_t D = c.kda_head_dim, DI = D * H, K = c.ssm_d_conv;
            KdaWeights& k = w.kda;
            k.wq = get(n("attn_q.weight"), {E, DI});
            k.wk = get(n("attn_k.weight"), {E, DI});
            k.wv = get(n("attn_v.weight"), {E, DI});
            k.conv_q = get(n("ssm_conv1d_q.weight"), {K, 1, DI});
            k.conv_k = get(n("ssm_conv1d_k.weight"), {K, 1, DI});
            k.conv_v = get(n("ssm_conv1d_v.weight"), {K, 1, DI});
            k.f_a = get(n("ssm_f_a.weight"), {E, D});
            k.f_b = get(n("ssm_f_b.weight"), {D, DI});
            k.beta = get(n("ssm_beta.weight"), {E, H});
            k.a = get(n("ssm_a"), {H});
            k.dt_b = get(n("ssm_dt.bias"), {DI});
            k.g_a = get(n("ssm_g_a.weight"), {E, D});
            k.g_b = get(n("ssm_g_b.weight"), {D, DI});
            k.o_norm = get(n("ssm_norm.weight"), {D});
            w.wo = get(n("attn_output.weight"), {DI, E});
        } else {
            const int64_t QL = c.q_lora_rank, KL = c.kv_lora_rank, HK = c.head_k_mla, HV = c.head_v_mla;
            MlaWeights& m = w.mla;
            m.q_a = get(n("attn_q_a.weight"), {E, QL});
            m.q_a_norm = get(n("attn_q_a_norm.weight"), {QL});
            m.q_b = get(n("attn_q_b.weight"), {QL, H * HK});
            m.kv_a = get(n("attn_kv_a_mqa.weight"), {E, KL + c.n_rot});
            m.kv_a_norm = get(n("attn_kv_a_norm.weight"), {KL});
            m.k_b = get(n("attn_k_b.weight"), {HK - c.n_rot, KL, H});
            m.v_b = get(n("attn_v_b.weight"), {KL, HV, H});
            w.wo = get(n("attn_output.weight"), {H * HV, E});
            const bool full = c.indexer_full[(size_t) il] != 0;
            const int64_t IH = c.indexer_n_head, ID = c.indexer_head_size;
            m.idx_k_norm = get(n("indexer.k_norm.weight"), {ID}, full);
            m.idx_k_norm_b = get(n("indexer.k_norm.bias"), {ID}, full);
            m.idx_proj = get(n("indexer.proj.weight"), {E, IH}, full);
            m.idx_attn_k = get(n("indexer.attn_k.weight"), {E, ID}, full);
            m.idx_attn_q_b = get(n("indexer.attn_q_b.weight"), {QL, IH * ID}, full);
            m.idx_kpool_gate = get(n("indexer_compressor_gate.weight"), {E, ID}, full);
            m.idx_kpool_ape = get(n("indexer_compressor_ape.weight"), {ID, c.indexer_kpool}, full);
        }

        if (c.is_dense(il)) {
            w.ffn_gate = get(n("ffn_gate.weight"), {E, c.n_ff});
            w.ffn_up = get(n("ffn_up.weight"), {E, c.n_ff});
            w.ffn_down = get(n("ffn_down.weight"), {c.n_ff, E});
        } else {
            const int64_t F = c.n_ff_exp[(size_t) il], NE = c.n_expert, FS = F * c.n_expert_shared;
            w.gate_inp = get(n("ffn_gate_inp.weight"), {E, NE});
            w.exp_probs_b = get(n("exp_probs_b.bias"), {NE});
            w.gate_exps = get(n("ffn_gate_exps.weight"), {E, F, NE});
            w.up_exps = get(n("ffn_up_exps.weight"), {E, F, NE});
            w.down_exps = get(n("ffn_down_exps.weight"), {F, E, NE});
            if (c.n_expert_shared > 0) {
                w.gate_shexp = get(n("ffn_gate_shexp.weight"), {E, FS});
                w.up_shexp = get(n("ffn_up_shexp.weight"), {E, FS});
                w.down_shexp = get(n("ffn_down_shexp.weight"), {FS, E});
            }
        }
    }
}

namespace {

struct Range {
    const uint8_t* p;
    size_t n;
};

std::vector<Range> expert_ranges(const Model& m) {
    std::vector<Range> out;
    const Config& c = m.config();
    for (int64_t il = 0; il < c.n_layer; ++il) {
        if (c.is_dense(il)) continue;
        const LayerWeights& w = m.layer(il);
        for (const Tensor* t : {&w.gate_exps, &w.up_exps, &w.down_exps})
            if (t->ok()) out.push_back({static_cast<const uint8_t*>(t->data), t->row_bytes() * (size_t) t->rows()});
    }
    return out;
}

}  // namespace

size_t Model::expert_bytes() const {
    size_t n = 0;
    for (const Range& r : expert_ranges(*this)) n += r.n;
    return n;
}

void Model::move_experts(int64_t il, const void* gate, const void* up, const void* down) {
    LayerWeights& w = layers_.at((size_t) il);
    w.gate_exps.data = gate;
    w.up_exps.data = up;
    w.down_exps.data = down;
}

void Model::warm_experts(int threads, const std::function<void(size_t, size_t)>& progress) const {
    // pieces of 64 MiB, taken by the threads in file order (the disk reads ahead along each)
    const size_t piece = (size_t) 64 << 20;
    std::vector<Range> pieces;
    size_t total = 0;
    for (const Range& r : expert_ranges(*this)) {
        total += r.n;
        for (size_t o = 0; o < r.n; o += piece) pieces.push_back({r.p + o, std::min(piece, r.n - o)});
    }
    if (pieces.empty()) return;
    const size_t page = (size_t) std::max(4096L, sysconf(_SC_PAGESIZE));
    std::atomic<size_t> next{0}, done{0};
    auto work = [&] {
        for (;;) {
            const size_t i = next.fetch_add(1);
            if (i >= pieces.size()) return;
            const Range& r = pieces[i];
            const uintptr_t a = (uintptr_t) r.p & ~(uintptr_t) (page - 1);
            madvise((void*) a, r.n + ((uintptr_t) r.p - a), MADV_WILLNEED);
            volatile uint8_t sink = 0;
            for (size_t o = 0; o < r.n; o += page) sink = sink ^ r.p[o];
            (void) sink;
            done.fetch_add(r.n);
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
}

}  // namespace strata::glm
