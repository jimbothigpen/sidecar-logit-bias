# sidecar-logit-bias

Out-of-tree sidecar handler plugin for the hub [`llama.cpp` fork](https://github.com/jimbothigpen/llama.cpp) (`jimbothigpen/llama.cpp`, 'llama-yggdrasil'):
token-level additive logit bias applied after the LM head, just before the
user reads logits.

The handler reads a list of token IDs and corresponding bias values from a
GGUF and adds them to the logits buffer once per generated batch. Useful
for hard-suppressing or boosting specific tokens (refusal phrases,
unwanted-language token ranges, EOS suppression for long-form generation,
etc.) — it complements activation-level steering like abliteration.

## Build

Requires the hub `llama.cpp` fork built + installed with the sidecar-plugin ABI
(the `--sidecar-load-plugin` loader landed in commit `5ba111253e`).

```bash
cmake -S . -B build -DLLAMA_INSTALL_PREFIX=/opt/llama-yggdrasil-vulkan
cmake --build build
```

Output: `build/libsidecar_logit_bias.so`.

## Use

```bash
LD_LIBRARY_PATH=/opt/llama-yggdrasil-vulkan/lib \
/opt/llama-yggdrasil-vulkan/bin/llama-cli \
  --sidecar-load-plugin /path/to/libsidecar_logit_bias.so \
  --sidecar-vectors /path/to/your.bias.gguf \
  -m model.gguf -p "..."
```

Chain with other sidecars by comma-separating paths in `--sidecar-vectors`:

```bash
--sidecar-load-plugin /path/to/libsidecar_logit_bias.so \
--sidecar-vectors model.abl.gguf,model.bias.gguf
```

## Producers (`tools/`)

`tools/write.py` — emit a bias GGUF from explicit token-id pairs:

```bash
python tools/write.py --pair 15043:-5.0 --pair 220:-2.5 --output out.bias.gguf
```

`tools/from_phrases.py` — tokenize phrase strings against a model GGUF and
emit a bias for the resulting token ids:

```bash
python tools/from_phrases.py \
  --model your-model.gguf \
  --phrases tools/anti-refusal.txt \
  --bias -5.0 \
  --output your-model.anti-refusal.bias.gguf
```

`tools/anti-refusal.txt` — starter phrase list (20 common refusal
prefixes) intended as a logit-level complement to activation-level
abliteration.

Both tools rely on `gguf-py` from the engine source tree (or a `pip install gguf`).
Set `FRANKENTURBO2_DIR=/path/to/frankenturbo2` to point them at the engine
clone if `import gguf` fails.

## On-disk schema

```
sidecar.type            str    "logit_bias"
logit_bias.token_ids    i32[n] vocabulary token ids to bias
logit_bias.values       f32[n] additive bias per token; can be negative
```

`token_ids` and `values` must have the same length. Out-of-range token ids
(`>= n_vocab` of the loaded model) are silently skipped at apply time.

Tokens are model-specific: never share a `.bias.gguf` produced for one
model family with another.
