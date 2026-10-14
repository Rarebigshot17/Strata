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

int main(int argc, char ** argv) {
    // Take our own flags out of argv; everything else goes to llama.cpp's parser.
    std::string out_path;
    std::string ids_out_path;
    std::string tokens_path;
    std::vector<char *> rest = { argv[0] };
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if ((a == "--out" || a == "--ids-out" || a == "--tokens-file") && i + 1 < argc) {
            (a == "--out" ? out_path : a == "--ids-out" ? ids_out_path : tokens_path) = argv[++i];
        } else {
            rest.push_back(argv[i]);
        }
    }
    if (out_path.empty()) {
        fprintf(stderr, "usage: %s -m MODEL.gguf (-p TEXT | -f FILE | --tokens-file IDS) --out LOGITS.bin "
                        "[--ids-out IDS.txt] [llama.cpp flags]\n", argv[0]);
        return 2;
    }

    common_params params;
    common_init();
    if (!common_params_parse((int) rest.size(), rest.data(), params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
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
        LOG_INF("positions %d-%d done\n", off, off + n - 1);
    }
    fclose(out);

    if (!ids_out_path.empty()) {
        std::ofstream f(ids_out_path);
        for (size_t i = 0; i < ids.size(); ++i) {
            f << ids[i] << (i + 1 < ids.size() ? "," : "\n");
        }
    }
    LOG_INF("wrote %zu rows of %d logits to %s\n", ids.size(), n_vocab, out_path.c_str());
    return 0;
}
