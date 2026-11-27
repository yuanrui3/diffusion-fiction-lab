"""Does LLaDA fix copied characters first? Commit order against copying from the summary.

Usage:
    python explore/copy_first.py      # writes explore/out/copy_first.md

Post hoc (PROTOCOL.md entry 36). Adaptive samplers fix the "easiest" positions
first (Kim et al. 2025). If copying from the summary is easy for a masked
model, tokens whose characters all occur in the summary should be fixed earlier
than tokens that are new. For the with-summary samples of the diffusion arms:
each line token's rank in the line's commit order (0 = first, 1 = last), split
by whether all its 汉字 occur in the summary, and whether they occur in the
given line.
"""

from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dfl.text import hanzi_only  # noqa: E402

OUT = ROOT / "explore" / "out"
END = {"llada": {126081, 126348}, "dream": {151643, 151645}}
NAMES = {"llada": "LLaDA 0.7", "llada-default": "LLaDA 0", "dream": "Dream 0.5", "dream-default": "Dream 0.2"}


def main() -> int:
    from transformers import AutoTokenizer
    from dfl.registry import MODELS, local_path

    items = {}
    for f in ["data/unseen/huohou.jsonl", "data/recall/recall.jsonl"]:
        for line in open(ROOT / f, encoding="utf-8"):
            it = json.loads(line)
            items[it["id"]] = it
    out = ["# Commit order against copying\n",
           "Post hoc; see `PROTOCOL.md` entry 36. With-summary samples, both sets. Each token of the line gets its rank in the line's commit order "
           "(0 = fixed first among the line's tokens, 1 = last). Tokens are split by whether every 汉字 in them occurs in the chapter summary. "
           "Tokens with no 汉字 are left out.\n",
           "| Sampler | Tokens from the summary | Their mean rank | Other tokens | Their mean rank | Difference |",
           "|---|---|---|---|---|---|"]
    toks = {}
    results = []
    for arm in ("llada", "llada-default", "dream", "dream-default"):
        base = "llada" if arm.startswith("llada") else "dream"
        if base not in toks:
            spec = MODELS[base]
            toks[base] = AutoTokenizer.from_pretrained(str(local_path(spec)), trust_remote_code=True)
        tok = toks[base]
        cache = {}
        copied, fresh = [], []
        for f in sorted(glob.glob(str(ROOT / "runs" / "full" / f"{arm}_*_summary.jsonl"))):
            hdr = None
            for line in open(f, encoding="utf-8"):
                rec = json.loads(line)
                if rec.get("type") == "header":
                    hdr = rec
                    continue
                if hdr["label"] != arm:
                    continue
                summ = items[rec["item_id"]].get("summary") or ""
                if not summ:
                    continue
                for s in rec["samples"]:
                    ids, steps = s.get("token_ids"), s.get("commit_step")
                    if not ids or not steps:
                        continue
                    L = next((j for j, t in enumerate(ids) if t in END[base]), len(ids))
                    if L < 2 or L == len(ids):
                        continue
                    ranks = np.argsort(np.argsort(steps[:L], kind="stable"), kind="stable") / (L - 1)
                    for j in range(L):
                        t = ids[j]
                        if t not in cache:
                            cache[t] = hanzi_only(tok.decode([t]))
                        h = cache[t]
                        if not h:
                            continue
                        pos_bin = min(2, int(3 * j / L))  # where in the line the token sits: first, middle or last third
                        (copied if all(c in summ for c in h) else fresh).append((ranks[j], pos_bin))
        c = np.array([x for x, _ in copied])
        r = np.array([x for x, _ in fresh])
        # the same difference within each third of the line, then averaged: copied tokens may simply sit early
        within = []
        for b in range(3):
            cb = [x for x, p in copied if p == b]
            rb = [x for x, p in fresh if p == b]
            if cb and rb:
                within.append(np.mean(cb) - np.mean(rb))
        results.append((arm, len(c), c.mean(), len(r), r.mean(), c.mean() - r.mean(), float(np.mean(within))))
        out.append(f"| {NAMES[arm]} | {len(c)} | {c.mean():.2f} | {len(r)} | {r.mean():.2f} | {c.mean() - r.mean():+.2f} |")
    out.append("")
    out.append("Controlled for position (the difference computed within the first, middle and last third of the line, then averaged):\n")
    out.append("| Sampler | Difference within position |")
    out.append("|---|---|")
    for arm, *_, w in results:
        out.append(f"| {NAMES[arm]} | {w:+.2f} |")
    out.append("")
    (OUT / "copy_first.md").write_text("\n".join(out) + "\n", encoding="utf-8", newline="\n")
    print("\n".join(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
