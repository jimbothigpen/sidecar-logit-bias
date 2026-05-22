#!/usr/bin/env python3
"""Build a logit_bias sidecar GGUF from phrase strings tokenized against a
specific model.

Pipeline:
    1. For each non-comment line in --phrases FILE, run `llama-tokenize` to
       resolve the phrase to a list of token IDs (model-specific).
    2. Collect unique token IDs across all phrases.
    3. Emit a logit_bias sidecar GGUF assigning the user-supplied --bias to
       each token ID.

The model.gguf MUST match (same tokenizer family) the model the sidecar
will be applied to at inference time. Otherwise the token IDs will not
correspond.

Schema is the same as tools/sidecar-logit-bias/write.py:
    sidecar.type            str    "logit_bias"
    logit_bias.token_ids    i32[n]
    logit_bias.values       f32[n]
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np


def tokenize_phrase(tokenize_binary: Path, model: Path, phrase: str,
                    no_bos: bool, parse_special: bool) -> list[int]:
    args = [str(tokenize_binary), "-m", str(model), "--ids", "-p", phrase, "--log-disable"]
    if no_bos:
        args.append("--no-bos")
    if not parse_special:
        args.append("--no-parse-special")
    out = subprocess.run(args, capture_output=True, text=True, check=True)
    text = out.stdout.strip()
    # llama-tokenize prints "[1, 2, 3]" with --ids
    m = re.search(r"\[(.*?)\]", text)
    if not m:
        raise RuntimeError(f"could not parse tokenize output: {text!r}")
    if not m.group(1).strip():
        return []
    return [int(x) for x in m.group(1).split(",") if x.strip()]


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, type=Path,
                    help="path to model GGUF (used only for its tokenizer)")
    ap.add_argument("--phrases", required=True, type=Path,
                    help="text file: one phrase per line; '#' and blank lines ignored")
    ap.add_argument("--bias", required=True, type=float,
                    help="additive bias applied to every collected token id "
                         "(use negative to suppress, positive to boost)")
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--tokenize-binary", type=Path,
                    default=Path("/opt/llama-frankenturbo2-rocm/bin/llama-tokenize"),
                    help="path to llama-tokenize")
    ap.add_argument("--ld-library-path", type=str,
                    default="/opt/llama-frankenturbo2-rocm/lib",
                    help="LD_LIBRARY_PATH for the tokenize subprocess")
    ap.add_argument("--with-bos", action="store_true",
                    help="include the BOS token in tokenization (default: omit)")
    ap.add_argument("--no-special", action="store_true",
                    help="skip control-token parsing (--no-parse-special)")
    ap.add_argument("--also-emit-pairs", type=Path, default=None,
                    help="optionally write a human-readable mapping "
                         "<token_id>\\t<phrase>\\t<bias> to this path")
    args = ap.parse_args()

    if not args.tokenize_binary.exists():
        print(f"ERROR: tokenize binary not found at {args.tokenize_binary}", file=sys.stderr)
        return 2
    if not args.model.exists():
        print(f"ERROR: model not found at {args.model}", file=sys.stderr)
        return 2

    # Pre-set the LD_LIBRARY_PATH for any subprocess.
    import os
    env_ld = os.environ.copy()
    if args.ld_library_path:
        env_ld["LD_LIBRARY_PATH"] = args.ld_library_path
    os.environ.update(env_ld)  # propagate so subprocess inherits

    raw_lines = args.phrases.read_text().splitlines()
    phrases = [s.strip() for s in raw_lines if s.strip() and not s.lstrip().startswith("#")]
    if not phrases:
        print(f"ERROR: no phrases in {args.phrases}", file=sys.stderr)
        return 2

    by_id: dict[int, str] = {}
    for phrase in phrases:
        try:
            ids = tokenize_phrase(
                args.tokenize_binary, args.model, phrase,
                no_bos=not args.with_bos,
                parse_special=not args.no_special,
            )
        except subprocess.CalledProcessError as e:
            print(f"ERROR: llama-tokenize failed on phrase {phrase!r}:\n{e.stderr}", file=sys.stderr)
            return 2
        for tid in ids:
            # First phrase that hits a token wins for the human-readable trace.
            by_id.setdefault(tid, phrase)

    if not by_id:
        print("ERROR: no token ids collected from phrases", file=sys.stderr)
        return 2

    token_ids = np.asarray(sorted(by_id.keys()), dtype=np.int32)
    values    = np.full(token_ids.shape, args.bias, dtype=np.float32)

    try:
        from gguf import GGUFWriter
    except ImportError:
        engine = os.environ.get("FRANKENTURBO2_DIR", "/mnt/cephfs/0/Container/systems/ai00/users/builduser/projects/frankenturbo2/src/jimbothigpen/frankenturbo2")
        sys.path.insert(0, f"{engine}/gguf-py")
        from gguf import GGUFWriter

    args.output.parent.mkdir(parents=True, exist_ok=True)
    w = GGUFWriter(str(args.output), "logit_bias")
    w.add_string("sidecar.type", "logit_bias")
    w.add_array("logit_bias.token_ids", token_ids.tolist())
    w.add_array("logit_bias.values",    values.tolist())
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.close()

    print(f"Wrote logit_bias sidecar with {len(token_ids)} unique token(s) "
          f"(bias={args.bias}) to {args.output}", file=sys.stderr)

    if args.also_emit_pairs is not None:
        with args.also_emit_pairs.open("w") as fh:
            for tid in sorted(by_id):
                fh.write(f"{tid}\t{by_id[tid]!r}\t{args.bias}\n")
        print(f"Also wrote pairs trace to {args.also_emit_pairs}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
