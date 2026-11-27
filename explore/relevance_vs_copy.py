"""Is the diffusion models' relevance edge explained by copying from the summary?

Usage:
    python explore/relevance_vs_copy.py     # writes explore/out/relevance_vs_copy.md

Post hoc (PROTOCOL.md entry 36). With-summary samples, unseen set (where the
phase 1 relevance difference was reported) and both sets together. Each
answer's copy share = share of its 汉字 found in the summary. The model
difference in relevance is computed overall and within copy-share bands
(stratified by the pooled quintiles), then averaged with the pooled band
weights. If the edge comes from copying, the within-band difference should be
much smaller than the overall one. Intervals: bootstrap over items.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dfl.text import hanzi_only  # noqa: E402

OUT = ROOT / "explore" / "out"
NAMES = {"qwen": "Qwen", "llada": "LLaDA 0.7", "dream": "Dream 0.5", "llada-default": "LLaDA 0", "dream-default": "Dream 0.2"}


def main() -> int:
    items = {}
    for f in ["data/unseen/huohou.jsonl", "data/recall/recall.jsonl"]:
        for line in open(ROOT / f, encoding="utf-8"):
            it = json.loads(line)
            items[it["id"]] = it
    rows = []
    with open(ROOT / "results" / "full_scored.jsonl", encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            if r["condition"] != "summary" or r.get("relevance") is None:
                continue
            summ = items[r["item_id"]].get("summary") or ""
            ans = hanzi_only(r["answer"] or "")
            if not ans or not summ:
                continue
            rows.append({"model": r["model"], "set": r["set"], "item": r["item_id"],
                         "copy": float(np.mean([c in summ for c in ans])), "rel": float(r["relevance"])})
    out = ["# Relevance against copying from the summary\n",
           "Post hoc; see `PROTOCOL.md` entry 36. With-summary answers. *Overall*: the model's mean relevance minus Qwen's. "
           "*Same copy share*: the same difference computed within five bands of copy share (pooled quintiles) and averaged with the pooled band weights. "
           "95% bootstrap intervals over items.\n"]
    rng = np.random.default_rng(20261125)
    for sets, label in ((("unseen",), "Unseen set"), (("unseen", "recall"), "Both sets")):
        sub = [r for r in rows if r["set"] in sets]
        edges = np.quantile([r["copy"] for r in sub], [0.2, 0.4, 0.6, 0.8])
        for r in sub:
            r["band"] = int(np.searchsorted(edges, r["copy"], side="right"))
        weights = np.bincount([r["band"] for r in sub], minlength=5) / len(sub)
        by_item = defaultdict(lambda: defaultdict(list))
        for r in sub:
            by_item[r["item"]][r["model"]].append(r)
        ids = sorted(by_item)

        def diffs(pick, m):
            allm = [r for i in pick for r in by_item[i][m]]
            allq = [r for i in pick for r in by_item[i]["qwen"]]
            if not allm or not allq:
                return np.nan, np.nan
            overall = np.mean([r["rel"] for r in allm]) - np.mean([r["rel"] for r in allq])
            within, w = [], []
            for b in range(5):
                mb = [r["rel"] for r in allm if r["band"] == b]
                qb = [r["rel"] for r in allq if r["band"] == b]
                if mb and qb:
                    within.append(np.mean(mb) - np.mean(qb))
                    w.append(weights[b])
            return overall, (np.average(within, weights=w) if within else np.nan)

        out.append(f"## {label}\n")
        out.append("| Model − Qwen | Overall | Same copy share |")
        out.append("|---|---|---|")
        for m in ("llada", "llada-default", "dream", "dream-default"):
            o, s = diffs(ids, m)
            draws = []
            for _ in range(1000):
                pick = [ids[k] for k in rng.integers(0, len(ids), len(ids))]
                draws.append(diffs(pick, m))
            d = np.array(draws, dtype=float)
            lo_o, hi_o = np.nanpercentile(d[:, 0], [2.5, 97.5])
            lo_s, hi_s = np.nanpercentile(d[:, 1], [2.5, 97.5])
            out.append(f"| {NAMES[m]} | {o:+.3f} [{lo_o:+.3f}, {hi_o:+.3f}] | {s:+.3f} [{lo_s:+.3f}, {hi_s:+.3f}] |")
        out.append("")
    (OUT / "relevance_vs_copy.md").write_text("\n".join(out) + "\n", encoding="utf-8", newline="\n")
    print("\n".join(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
