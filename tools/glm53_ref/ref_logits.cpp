// Reference logits for the GLM-5.3-Flash port, computed by llama.cpp.
//
// The port is checked by running the SAME token ids through llama.cpp and through Strata and comparing the
// logits at every position (tools/glm53_compare.py). llama.cpp's own `llama-debug --save-logits` only keeps the
// last position, so this small tool keeps all of them.
//
// Output files:
//   --out PATH      logits in the layout `strata generate --dump-logits` writes:
//                   int32 n_vocab, int32 n_rows, then n_rows x n_vocab float32, one row per position
//   --ids-out PATH  the token ids, comma separated, ready for `strata generate --tokens-file`
//   --dump-layers PATH  per position, for every layer, the graph's tensors that --dump-names lists (default
//                   hc_attn_post,l_out: the residual streams after the attention and after the FFN, hc x n_embd
//                   float32 each), in that order, the layout `strata-glm-ref --dump-layers` writes with the same
//                   --dump-names (tools/glm53_layers_compare.py compares the two). Needs -b 1 -ub 1, one position per
//                   decode. Other names: hc_attn_pre, attn_norm, hc_ffn_pre, ffn_norm, ffn_out (n_embd each).
//
// Input is either a prompt (-p TEXT or -f FILE, tokenized by llama.cpp) or --tokens-file PATH (ids separated by
// commas or whitespace). Every other flag is llama.cpp's own (-m, -ngl, --n-cpu-moe, -c, -t, ...).
//
// Built inside a llama.cpp checkout by tools/glm53_ref/build.sh.

#include "arg.h"
#include "common.h"
#include "llama.h"
#include "log.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

static bool read_ids(const std::string & path, std::vector<llama_token> & out) {
    std::ifstream in(path);
    if (!in) {
        return false;
    }
    std::stringstream ss;
    ss << in.rdbuf();
    std::string text = ss.str();
    for (char & c : text) {
        if (c == ',') {
            c = ' ';
        }
    }
    std::istringstream is(text);
    long long v;
    while (is >> v) {
        out.push_back((llama_token) v);
    }
    return is.eof();
}

// --dump-layers: the current position's named tensors per layer, filled by the scheduler's eval callback
struct layer_dump {
    int32_t n_layer = 0;
    std::vector<std::string> names = { "hc_attn_post", "l_out" };
    std::vector<std::vector<std::vector<float>>> data;   // [name][layer]
    bool bad = false;                                    // a tensor that is not contiguous f32
};

static bool layer_dump_cb(ggml_tensor * t, bool ask, void * user_data) {
    auto * d = (layer_dump *) user_data;
    // the graph names a layer's tensor "<name>-<layer>"
    const char * dash = strrchr(t->name, '-');
    int which = -1;
    int il = -1;
    if (dash != nullptr && dash[1] != 0 && strspn(dash + 1, "0123456789") == strlen(dash + 1)) {
        const std::string base(t->name, (size_t) (dash - t->name));
        for (size_t k = 0; k < d->names.size(); ++k) {
            if (d->names[k] == base) {
                which = (int) k;
            }
        }
        il = atoi(dash + 1);
    }
    if (ask) {
        return which >= 0;
    }
    if (which < 0 || il < 0 || il >= d->n_layer) {
        return true;
    }
    if (t->type != GGML_TYPE_F32 || !ggml_is_contiguous(t)) {
        d->bad = true;
        return true;
    }
    auto & v = d->data[(size_t) which][(size_t) il];
    v.resize((size_t) ggml_nelements(t));
    ggml_backend_tensor_get(t, v.data(), 0, ggml_nbytes(t));
    return true;
}

int main(int argc, char ** argv) {
    // Take our own flags out of argv; everything else goes to llama.cpp's parser.
    std::string out_path;
    std::string ids_out_path;
    std::string tokens_path;
    std::string layers_path;
    std::string dump_names;
    std::vector<char *> rest = { argv[0] };
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if ((a == "--out" || a == "--ids-out" || a == "--tokens-file") && i + 1 < argc) {
            (a == "--out" ? out_path : a == "--ids-out" ? ids_out_path : tokens_path) = argv[++i];
        } else if (a == "--dump-layers" && i + 1 < argc) {
            layers_path = argv[++i];
        } else if (a == "--dump-names" && i + 1 < argc) {
            dump_names = argv[++i];
        } else {
            rest.push_back(argv[i]);
        }
    }
    if (out_path.empty()) {
        fprintf(stderr, "usage: %s -m MODEL.gguf (-p TEXT | -f FILE | --tokens-file IDS) --out LOGITS.bin "
                        "[--ids-out IDS.txt] [--dump-layers LAYERS.bin [--dump-names a,b]] [llama.cpp flags]\n",
                argv[0]);
        return 2;
    }

    common_params params;
    common_init();
    if (!common_params_parse((int) rest.size(), rest.data(), params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }
    layer_dump dump;
    if (!dump_names.empty()) {
        dump.names.clear();
        std::stringstream ns(dump_names);
        for (std::string n; std::getline(ns, n, ',');) {
            if (!n.empty()) {
                dump.names.push_back(n);
            }
        }
    }
    if (!layers_path.empty()) {
        params.cb_eval = layer_dump_cb;
        params.cb_eval_user_data = &dump;
    }
    llama_backend_init();
    llama_numa_init(params.numa);

    auto init = common_init_from_params(params);
    llama_model * model = init->model();
    llama_context * ctx = init->context();
    if (model == nullptr || ctx == nullptr) {
        LOG_ERR("failed to load the model\n");
        return 1;
    }
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int32_t n_vocab = llama_vocab_n_tokens(vocab);

    std::vector<llama_token> ids;
    if (!tokens_path.empty()) {
        if (!read_ids(tokens_path, ids)) {
            LOG_ERR("cannot read token ids from %s\n", tokens_path.c_str());
            return 1;
        }
    } else {
        ids = common_tokenize(ctx, params.prompt, llama_vocab_get_add_bos(vocab), true);
    }
    if (ids.empty()) {
        LOG_ERR("no input tokens: pass -p, -f or --tokens-file\n");
        return 1;
    }
    for (llama_token id : ids) {
        if (id < 0 || id >= n_vocab) {
            LOG_ERR("token id %d is outside the vocab (%d)\n", id, n_vocab);
            return 1;
        }
    }
    if ((int64_t) ids.size() > (int64_t) llama_n_ctx(ctx)) {
        LOG_ERR("%zu tokens do not fit in the context (%u); raise -c\n", ids.size(), llama_n_ctx(ctx));
        return 1;
    }

    FILE * out = fopen(out_path.c_str(), "wb");
    if (out == nullptr) {
        LOG_ERR("cannot write %s\n", out_path.c_str());
        return 1;
    }
    const int32_t hdr[2] = { n_vocab, (int32_t) ids.size() };
    fwrite(hdr, sizeof(hdr), 1, out);

    FILE * lay = nullptr;
    if (!layers_path.empty()) {
        if (llama_n_batch(ctx) != 1 || llama_n_ubatch(ctx) != 1) {
            LOG_ERR("--dump-layers needs -b 1 -ub 1 (one position per decode)\n");
            fclose(out);
            return 1;
        }
        dump.n_layer = llama_model_n_layer(model);
        dump.data.assign(dump.names.size(), std::vector<std::vector<float>>((size_t) dump.n_layer));
        lay = fopen(layers_path.c_str(), "wb");
        if (lay == nullptr) {
            LOG_ERR("cannot write %s\n", layers_path.c_str());
            fclose(out);
            return 1;
        }
    }
    int32_t dumped = -1;   // the layers each position had, the same for every position

    // Decode in chunks of n_batch with an output on every position, writing each chunk's rows as it finishes.
    const int32_t n_batch = (int32_t) llama_n_batch(ctx);
    for (int32_t off = 0; off < (int32_t) ids.size(); off += n_batch) {
        const int32_t n = std::min<int32_t>(n_batch, (int32_t) ids.size() - off);
        common_batch batch(ctx);
        for (int32_t j = 0; j < n; ++j) {
            batch.add(ids[off + j], off + j, 0, true);
        }
        if (llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch.get()) != 0) {
            LOG_ERR("decode failed at position %d\n", off);
            fclose(out);
            return 1;
        }
        for (int32_t j = 0; j < n; ++j) {
            const float * row = llama_get_logits_ith(ctx, j);
            if (row == nullptr || fwrite(row, sizeof(float), (size_t) n_vocab, out) != (size_t) n_vocab) {
                LOG_ERR("cannot read or write the logits of position %d\n", off + j);
                fclose(out);
                return 1;
            }
        }
        if (lay != nullptr) {
            // the layers the graph ran (the model's count can include an MTP block it does not run), in order
            const auto layer_done = [&](int32_t il) {
                for (const auto & per_name : dump.data) {
                    if (per_name[(size_t) il].empty()) {
                        return false;
                    }
                }
                return true;
            };
            int32_t have = 0;
            while (have < dump.n_layer && layer_done(have)) {
                ++have;
            }
            if (dump.bad || have == 0 || (dumped >= 0 && have != dumped)) {
                LOG_ERR("--dump-layers: position %d gave %d layers (%s)\n", off, have,
                        dump.bad ? "a dumped tensor was not contiguous f32" :
                        have == 0 ? "a name in --dump-names is not in the graph" : "expected the same count every time");
                fclose(out);
                fclose(lay);
                return 1;
            }
            dumped = have;
            for (int32_t il = 0; il < have; ++il) {
                for (auto & per_name : dump.data) {
                    auto & v = per_name[(size_t) il];
                    fwrite(v.data(), sizeof(float), v.size(), lay);
                    v.clear();
                }
            }
        }
        LOG_INF("positions %d-%d done\n", off, off + n - 1);
    }
    fclose(out);
    if (lay != nullptr) {
        fclose(lay);
        LOG_INF("wrote %d layers' %s per position to %s\n", dumped, dump_names.empty() ? "hc_attn_post,l_out" :
                dump_names.c_str(), layers_path.c_str());
    }

    if (!ids_out_path.empty()) {
        std::ofstream f(ids_out_path);
        for (size_t i = 0; i < ids.size(); ++i) {
            f << ids[i] << (i + 1 < ids.size() ? "," : "\n");
        }
    }
    LOG_INF("wrote %zu rows of %d logits to %s\n", ids.size(), n_vocab, out_path.c_str());
    return 0;
}
