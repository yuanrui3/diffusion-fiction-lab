"""What if phase 1 had scored tone opposition instead of tone assignment? And do the grades track copying?

Usage:
    python explore/rescore_opposition.py     # writes explore/out/opposition_rescore.md

Post hoc (PROTOCOL.md entry 36). Not a replacement for the frozen results.

1. Re-scoring. The frozen "all rules" needs the right length, the end tone the
   convention assigns (first line 仄, second 平), no repeats, and no copy of the
   given line. The classical 回目 keep a different rule: the two lines end on
   different tones, whichever way round (explore/out/existing_data.md). Here
   "all rules (opposition)" swaps the assignment for opposition (modern tones),
   for every model and for the originals, on both sets. Intervals are bootstrap
   intervals over items (dfl.stats).
2. Grades and copying. For the 48 lines the author graded: the share of the
   line's 汉字 found in the chapter summary, against the grade.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dfl.checker import repeated_characters  # noqa: E402
from dfl.stats import diff_paired, mean_ci  # noqa: E402
from dfl.text import hanzi_only  # noqa: E402
from dfl.tones import PING, ZE, modern_classes  # noqa: E402

OUT = ROOT / "explore" / "out"
MODELS = ["qwen", "llada", "dream", "llada-default", "dream-default"]
NAMES = {"qwen": "Qwen", "llada": "LLaDA 0.7", "dream": "Dream 0.5", "llada-default": "LLaDA 0", "dream-default": "Dream 0.2"}


def end_class(line):
    h = hanzi_only(line)
    if not h:
        return None
    c = modern_classes(h[-1])[0]
    return c if c in (PING, ZE) else None


def fmt(r):
    return f"{100 * r['mean']:.0f}% [{100 * r['lo']:.0f}, {100 * r['hi']:.0f}]" if r.get("mean") is not None else "–"


def main() -> int:
    items = {}
    for f in ["data/unseen/huohou.jsonl", "data/recall/recall.jsonl"]:
        for line in open(ROOT / f, encoding="utf-8"):
            it = json.loads(line)
            items[it["id"]] = it
    frozen = defaultdict(lambda: defaultdict(list))
    opp = defaultdict(lambda: defaultdict(list))
    with open(ROOT / "results" / "full_scored.jsonl", encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            key = (r["model"], r["set"], r["condition"], r["direction"])
            frozen[key][r["item_id"]].append(1.0 if r["all_rules_modern"] else 0.0)
            g, e = end_class(r["given"]), r.get("end_modern_class")
            ok = (bool(r.get("rule_length")) and bool(r.get("rule_no_repeat")) and not r.get("echo")
                  and g is not None and e in (PING, ZE) and e != g)
            opp[key][r["item_id"]].append(1.0 if ok else 0.0)

    out = ["# Phase 1 re-scored with tone opposition\n",
           "Post hoc; see `PROTOCOL.md` entry 36. *Frozen*: the phase 1 all-rules rate (length, assigned end tone, no repeat, not a copy). "
           "*Opposition*: the same, with the assigned end tone replaced by \"the two lines end on different tones\" (modern tones; a line whose "
           "final tone is unknown fails). Bare condition, both directions pooled; 95% bootstrap intervals over items.\n",
           "| Writer | Unseen, frozen | Unseen, opposition | Recall, frozen | Recall, opposition |",
           "|---|---|---|---|---|"]
    # originals as the ceiling
    ceil = {}
    for set_name in ("unseen", "recall"):
        fz, op = {}, {}
        for i, it in items.items():
            if it["set"] != set_name:
                continue
            a, b = it["first"], it["second"]
            same_len = len(hanzi_only(a)) == len(hanzi_only(b))
            no_rep = not repeated_characters(hanzi_only(a), hanzi_only(b))  # the frozen rule: same-position repeats allowed
            ca, cb = end_class(a), end_class(b)
            fz[i] = [1.0 if (same_len and no_rep and ca == ZE and cb == PING) else 0.0]
            op[i] = [1.0 if (same_len and no_rep and ca and cb and ca != cb) else 0.0]
        ceil[set_name] = (mean_ci(fz), mean_ci(op))
    out.append(f"| Originals (both lines) | {fmt(ceil['unseen'][0])} | {fmt(ceil['unseen'][1])} | {fmt(ceil['recall'][0])} | {fmt(ceil['recall'][1])} |")
    data = {}
    for m in MODELS:
        cells = []
        for set_name in ("unseen", "recall"):
            fz, op = defaultdict(list), defaultdict(list)
            for d in ("forward", "reverse"):
                for i, v in frozen[(m, set_name, "bare", d)].items():
                    fz[i] += v
                for i, v in opp[(m, set_name, "bare", d)].items():
                    op[i] += v
            a, b = mean_ci(fz), mean_ci(op)
            data[f"{m}|{set_name}"] = {"frozen": a, "opposition": b}
            cells += [fmt(a), fmt(b)]
        out.append(f"| {NAMES[m]} | {cells[0]} | {cells[1]} | {cells[2]} | {cells[3]} |")
    out.append("")
    out.append("The originals' row is a ceiling for the rules only, with each couplet counted once; the frozen ceiling in "
               "`full_report.md` table 4 is computed per direction, so it differs slightly.\n")

    out.append("### LLaDA 0.7 − Qwen under each rule (bare, paired by item)\n")
    out.append("| Set | Direction | Frozen | Opposition |")
    out.append("|---|---|---|---|")
    for set_name in ("unseen", "recall"):
        for d in ("forward", "reverse"):
            a = diff_paired(frozen[("llada", set_name, "bare", d)], frozen[("qwen", set_name, "bare", d)])
            b = diff_paired(opp[("llada", set_name, "bare", d)], opp[("qwen", set_name, "bare", d)])
            f = lambda r: f"{100 * r['diff']:+.0f} [{100 * r['lo']:+.0f}, {100 * r['hi']:+.0f}]{' *' if (r['lo'] > 0 or r['hi'] < 0) else ''}"
            out.append(f"| {set_name} | {d} | {f(a)} | {f(b)} |")
    out.append("")

    # grades and copying
    gp = ROOT / "results" / "grading" / "graded.jsonl"
    if gp.exists():
        graded = [json.loads(l) for l in open(gp, encoding="utf-8")]
        xs, ys, ms = [], [], []
        for g in graded:
            if g["score"] is None:
                continue
            summ = items[g["item_id"]].get("summary") or ""
            ans = hanzi_only(g["answer"])
            if not ans or not summ:
                continue
            xs.append(np.mean([c in summ for c in ans]))
            ys.append(g["score"])
            ms.append(g["model"])
        xs, ys = np.array(xs), np.array(ys, dtype=float)
        from scipy.stats import spearmanr
        rho, p = spearmanr(xs, ys)
        out.append("### The author's grades against copying from the summary\n")
        out.append(f"{len(xs)} graded lines. Spearman correlation between the share of the line's 汉字 found in the summary and the grade: "
                   f"ρ = {rho:+.2f} (p = {p:.2f}).\n")
        out.append("| Model | Lines | Mean share from the summary | Mean grade |")
        out.append("|---|---|---|---|")
        for m in ("qwen", "llada", "dream"):
            sel = [k for k, mm in enumerate(ms) if mm == m]
            out.append(f"| {NAMES[m]} | {len(sel)} | {100 * xs[sel].mean():.0f}% | {ys[sel].mean():.2f} |")
        within = []
        for m in ("qwen", "llada", "dream"):
            sel = [k for k, mm in enumerate(ms) if mm == m]
            if len(sel) > 3:
                r_, p_ = spearmanr(xs[sel], ys[sel])
                within.append(f"{NAMES[m]} ρ = {r_:+.2f} (n = {len(sel)})")
        out.append("")
        out.append("Within each model: " + "; ".join(within) + ".\n")
    (OUT / "opposition_rescore.md").write_text("\n".join(out) + "\n", encoding="utf-8", newline="\n")
    (OUT / "opposition_rescore.json").write_text(json.dumps(data, indent=1, default=float) + "\n", encoding="utf-8", newline="\n")
    print("\n".join(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
