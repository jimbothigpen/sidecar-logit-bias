#!/usr/bin/env python3
"""Pack a token-id → bias mapping into a logit_bias sidecar GGUF.

Schema (matches src/sidecar/logit_bias.cpp):

    sidecar.type            str    "logit_bias"
    logit_bias.token_ids    i32[n] vocabulary token ids to bias
    logit_bias.values       f32[n] additive bias per token; can be negative

The handler iterates this array once per produced token batch and adds
`values[i]` to the `token_ids[i]`-th logit slot. Out-of-range token ids
(>= n_vocab of the target model) are silently skipped at apply time, so a
bias produced against a fine-tune with extra tokens still works against the
base model.

Inputs:
    --pairs FILE        text file, one "token_id<TAB>bias" pair per line.
                         '#' and blank lines are ignored. token_id is parsed
                         as int, bias as float.
    --pair  ID:VAL      may be repeated; equivalent to a single line in --pairs.
    --output FILE       output path (e.g. my_model.bias.gguf).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pairs", type=Path, default=None,
                    help="text file: one 'token_id<TAB>bias' pair per line")
    ap.add_argument("--pair", action="append", default=[],
                    help="ID:VAL pair (repeatable)")
    ap.add_argument("--output", required=True, type=Path)
    args = ap.parse_args()

    # Resolve gguf-py: prefer an installed `gguf` package; otherwise fall back
    # to the engine sibling clone at FRANKENTURBO2_DIR (env var) or
    # /usr/src/llama-forks/frankenturbo2/gguf-py if set.
    try:
        from gguf import GGUFWriter
    except ImportError:
        import os
        engine = os.environ.get("FRANKENTURBO2_DIR", "/usr/src/llama-forks/frankenturbo2")
        sys.path.insert(0, f"{engine}/gguf-py")
        from gguf import GGUFWriter

    pairs: list[tuple[int, float]] = []
    if args.pairs is not None:
        for raw in args.pairs.read_text().splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) != 2:
                print(f"ERROR: malformed line in {args.pairs}: {raw!r}", file=sys.stderr)
                return 2
            pairs.append((int(parts[0]), float(parts[1])))
    for spec in args.pair:
        if ":" not in spec:
            print(f"ERROR: --pair {spec!r} must be ID:VAL", file=sys.stderr)
            return 2
        tid, val = spec.split(":", 1)
        pairs.append((int(tid), float(val)))

    if not pairs:
        print("ERROR: no token-id/bias pairs supplied (--pairs FILE or --pair ID:VAL)",
              file=sys.stderr)
        return 2

    token_ids = np.asarray([p[0] for p in pairs], dtype=np.int32)
    values    = np.asarray([p[1] for p in pairs], dtype=np.float32)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Architecture string is mandatory in GGUF but unused by the logit_bias
    # handler — it doesn't validate against a model. Pass "logit_bias" so the
    # file self-identifies if a user inspects general.architecture.
    w = GGUFWriter(str(args.output), "logit_bias")
    w.add_string("sidecar.type", "logit_bias")
    w.add_array("logit_bias.token_ids", token_ids.tolist())
    w.add_array("logit_bias.values",    values.tolist())
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.close()

    print(f"Wrote logit_bias sidecar with {len(pairs)} entries to {args.output}",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
