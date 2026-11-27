"""Exact recall of classical 回目, counted by item and direction.

Usage:
    python explore/recall_direction.py      # writes explore/out/recall_direction.md

Post hoc (PROTOCOL.md entry 37). The frozen tables report exact match per
attempt, which rounds to 0–3% and hides which items were recalled and in which
direction. Here an item counts as recalled in a direction if any attempt, in
either condition, reproduced the original line exactly.
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

from scipy.stats import fisher_exact

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "explore" / "out"
MODELS = ["qwen", "llada", "llada-default", "dream", "dream-default"]
NAMES = {"qwen": "Qwen", "llada": "LLaDA 0.7", "dream": "Dream 0.5", "llada-default": "LLaDA 0", "dream-default": "Dream 0.2"}


def main() -> int:
    work = {}
    for line in open(ROOT / "data" / "recall" / "recall.jsonl", encoding="utf-8"):
        it = json.loads(line)
        work[it["id"]] = it["work"]
    items = defaultdict(set)
    attempts = Counter()
    total = Counter()
    for line in open(ROOT / "results" / "full_scored.jsonl", encoding="utf-8"):
        r = json.loads(line)
        if r["set"] != "recall":
            continue
        key = (r["model"], r["direction"])
        total[key] += 1
        if r["exact"]:
            items[key].add(r["item_id"])
            attempts[key] += 1
    n_items = len(work)
    out = ["# Exact recall by item and direction\n",
           "Post hoc; see `PROTOCOL.md` entry 37. An item counts as recalled in a direction if any attempt, with or without the summary, "
           f"reproduced the original line exactly. {n_items} classical items. *Forward*: given the first line, the original second line. "
           "*Reverse*: given the second line, the original first line. Fisher's exact test compares the two directions' item counts.\n",
           "| Model | Forward: items (attempts) | Reverse: items (attempts) | Forward / reverse | Fisher p |",
           "|---|---|---|---|---|"]
    for m in MODELS:
        f, b = len(items[(m, "forward")]), len(items[(m, "reverse")])
        p = fisher_exact([[f, n_items - f], [b, n_items - b]]).pvalue
        ratio = f"{f / b:.1f}" if b else ("–" if not f else "∞")
        out.append(f"| {NAMES[m]} | {f} ({attempts[(m, 'forward')]} of {total[(m, 'forward')]}) | {b} ({attempts[(m, 'reverse')]} of {total[(m, 'reverse')]}) | {ratio} | {p:.2g} |")
    q = (len(items[("qwen", "forward")]), len(items[("qwen", "reverse")]))
    l = (len(items[("llada", "forward")]), len(items[("llada", "reverse")]))
    p = fisher_exact([list(q), list(l)]).pvalue
    out.append("")
    out.append(f"Direction profile, Qwen against LLaDA 0.7, among recalled items ({q[0]}:{q[1]} against {l[0]}:{l[1]}): Fisher p = {p:.2g}.\n")
    out.append("Recalled items by work (Qwen, forward): " + ", ".join(f"{w} {c}" for w, c in Counter(work[i] for i in items[("qwen", "forward")]).most_common()) + ".\n")
    # partial recall: share of positions identical to the original, all items, both conditions
    import numpy as np
    same = defaultdict(lambda: defaultdict(list))
    for line in open(ROOT / "results" / "full_scored.jsonl", encoding="utf-8"):
        r = json.loads(line)
        if r["set"] == "recall" and r.get("ref_same_position") is not None:
            same[(r["model"], r["direction"])][r["item_id"]].append(r["ref_same_position"])
    rng = np.random.default_rng(20261125)
    ids = sorted(work)
    def item_means(m, d):
        return np.array([np.mean(same[(m, d)][i]) if same[(m, d)].get(i) else np.nan for i in ids])
    def ci(vec):
        vec = vec[~np.isnan(vec)]
        draws = [vec[rng.integers(0, len(vec), len(vec))].mean() for _ in range(4000)]
        return vec.mean(), np.percentile(draws, 2.5), np.percentile(draws, 97.5)
    out.append("## Partial recall\n")
    out.append("Share of positions where the answer has the original's character, averaged per item, over all 580 items (no selection), "
               "both conditions. 95% bootstrap intervals over items. A model that stores a couplet in one direction only should score "
               "higher forward than reverse; one that stores it symmetrically should score about the same.\n")
    out.append("| Model | Forward | Reverse | Forward − reverse |")
    out.append("|---|---|---|---|")
    for m in MODELS:
        f, b = item_means(m, "forward"), item_means(m, "reverse")
        cf, cb, cd = ci(f), ci(b), ci(f - b)
        out.append(f"| {NAMES[m]} | {100 * cf[0]:.1f}% | {100 * cb[0]:.1f}% | {100 * cd[0]:+.1f} [{100 * cd[1]:+.1f}, {100 * cd[2]:+.1f}] |")
    for d in ("forward", "reverse"):
        diff = item_means("llada", d) - item_means("qwen", d)
        c = ci(diff)
        out.append(f"\nLLaDA 0.7 − Qwen, {d}: {100 * c[0]:+.1f} points [{100 * c[1]:+.1f}, {100 * c[2]:+.1f}].")
    out.append("")
    (OUT / "recall_direction.md").write_text("\n".join(out) + "\n", encoding="utf-8", newline="\n")
    print("\n".join(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
