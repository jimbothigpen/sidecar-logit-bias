// Standalone out-of-tree sidecar plugin: token-level additive logit bias.
//
// Loaded into a frankenturbo2 / llama.cpp engine via:
//
//     llama-cli --sidecar-load-plugin /path/to/libsidecar_logit_bias.so \
//               --sidecar-vectors /path/to/your.bias.gguf
//
// On-disk schema (under the "logit_bias" GGUF KV namespace):
//   sidecar.type            str    "logit_bias"
//   logit_bias.token_ids    i32[n] vocabulary token ids to bias
//   logit_bias.values       f32[n] additive bias per token; can be negative
//
// `token_ids` and `values` must have the same length. Out-of-range token ids
// (>= n_vocab of the loaded model) are silently skipped at apply time, so a
// bias produced against a fine-tune with extra tokens still works against
// the base model.

#include <llama-sidecar-plugin.h>

#include <ggml.h>
#include <gguf.h>

#include <cstdarg>
#include <cstdio>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace {

inline void log_err(const char * fmt, ...) {
    va_list args;
    va_start(args, fmt);
    std::vfprintf(stderr, fmt, args);
    va_end(args);
}

inline void log_info(const char * fmt, ...) {
    va_list args;
    va_start(args, fmt);
    std::vfprintf(stderr, fmt, args);
    va_end(args);
}

struct logit_bias_handler : public llama_sidecar_handler {
    std::string type() const override { return "logit_bias"; }

    bool load(
            const llama_model & model,
            gguf_context * gguf,
            ggml_context * /*ctx_meta*/,
            const std::string & /*path*/,
            float /*scale_override*/,
            float /*threshold_override*/) override {
        const int id_tok = gguf_find_key(gguf, "logit_bias.token_ids");
        if (id_tok < 0) {
            log_err("logit_bias: missing required key 'logit_bias.token_ids'\n");
            return false;
        }
        if (gguf_get_kv_type(gguf, id_tok) != GGUF_TYPE_ARRAY ||
            gguf_get_arr_type(gguf, id_tok) != GGUF_TYPE_INT32) {
            log_err("logit_bias: 'logit_bias.token_ids' must be an int32 array\n");
            return false;
        }

        const int id_val = gguf_find_key(gguf, "logit_bias.values");
        if (id_val < 0) {
            log_err("logit_bias: missing required key 'logit_bias.values'\n");
            return false;
        }
        if (gguf_get_kv_type(gguf, id_val) != GGUF_TYPE_ARRAY ||
            gguf_get_arr_type(gguf, id_val) != GGUF_TYPE_FLOAT32) {
            log_err("logit_bias: 'logit_bias.values' must be a float32 array\n");
            return false;
        }

        const size_t n_tok = gguf_get_arr_n(gguf, id_tok);
        const size_t n_val = gguf_get_arr_n(gguf, id_val);
        if (n_tok != n_val) {
            log_err("logit_bias: token_ids (%zu) and values (%zu) must have same length\n",
                    n_tok, n_val);
            return false;
        }

        token_ids.assign((const int32_t *) gguf_get_arr_data(gguf, id_tok),
                         (const int32_t *) gguf_get_arr_data(gguf, id_tok) + n_tok);
        values.assign(   (const float   *) gguf_get_arr_data(gguf, id_val),
                         (const float   *) gguf_get_arr_data(gguf, id_val) + n_val);

        const int32_t n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(&model));
        size_t in_range = 0;
        for (size_t i = 0; i < token_ids.size(); ++i) {
            if (token_ids[i] >= 0 && token_ids[i] < n_vocab) ++in_range;
        }
        log_info("logit_bias: loaded %zu entries (%zu in-range against n_vocab=%d)\n",
                 token_ids.size(), in_range, n_vocab);
        return true;
    }

    void post_compute_logits(
            float * logits,
            int     n_vocab,
            int     n_tokens) const override {
        if (token_ids.empty() || logits == nullptr || n_tokens <= 0 || n_vocab <= 0) {
            return;
        }
        for (int t = 0; t < n_tokens; ++t) {
            float * row = logits + (size_t) t * (size_t) n_vocab;
            for (size_t i = 0; i < token_ids.size(); ++i) {
                const int32_t id = token_ids[i];
                if (id < 0 || id >= n_vocab) continue;
                row[id] += values[i];
            }
        }
    }

private:
    std::vector<int32_t> token_ids;
    std::vector<float>   values;
};

} // namespace

LLAMA_SIDECAR_PLUGIN_INIT_DECL {
    llama_sidecar_register(
        "logit_bias",
        []() -> llama_sidecar_handler_ptr {
            return std::make_shared<logit_bias_handler>();
        });
    return 0;
}
