"""Tables for the step-0 canvas probe (explore/probe_length.py).

Usage:
    python explore/report_length.py      # writes explore/out/length_probe.md
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "explore" / "out"
NAMES = {"llada": "LLaDA", "dream": "Dream"}


def main() -> int:
    out = ["# What the diffusion models expect before the first step\n",
           "Post hoc; see `PROTOCOL.md` entry 36 and `explore/probe_length.py`. One forward pass on the fully masked phase 1 canvas "
           "(required characters + 6 tokens), exact phase 1 prompt, bare, both directions, all 723 items.\n",
           "*Expected length*: the first canvas position whose probability of an end token exceeds 0.5. "
           "*Original's length*: the original line in the model's own tokens. "
           "*First pick*: the position each sampler would fix first: for LLaDA (low-confidence remasking) the most confident position, "
           "for Dream (entropy) the lowest-entropy position. *In the padding* means that position is predicted to be an end token. "
           "*Leftmost* means it is the first canvas position.\n",
           "| Model | Canvases | Expected length = original's | Mean expected − original (tokens) | First pick in the padding | First pick leftmost | p(end), line part | p(end), padding part |",
           "|---|---|---|---|---|---|---|---|"]
    for key in ("llada", "dream"):
        p = OUT / f"probe_L_{key}.jsonl"
        if not p.exists():
            continue
        rows = [json.loads(l) for l in open(p, encoding="utf-8")]
        exp_len, diffs, pad_first, left_first, in_line, in_pad = [], [], [], [], [], []
        for r in rows:
            pe = np.array(r["p_end"])
            over = np.nonzero(pe > 0.5)[0]
            L = int(over[0]) if len(over) else len(pe)
            exp_len.append(L)
            diffs.append(L - r["ref_tokens"])
            pick = int(np.argmax(r["conf"])) if key == "llada" else int(np.argmin(r["entropy"]))
            pad_first.append(pe[pick] > 0.5)
            left_first.append(pick == 0)
            if L > 0:
                in_line.append(pe[:L].mean())
            if L < len(pe):
                in_pad.append(pe[L:].mean())
        d = np.array(diffs)
        out.append(f"| {NAMES[key]} | {len(rows)} | {100 * np.mean(d == 0):.0f}% | {d.mean():+.2f} | {100 * np.mean(pad_first):.0f}% | "
                   f"{100 * np.mean(left_first):.0f}% | {np.mean(in_line):.2f} | {np.mean(in_pad):.2f} |")
    out.append("")
    (OUT / "length_probe.md").write_text("\n".join(out) + "\n", encoding="utf-8", newline="\n")
    print("\n".join(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
