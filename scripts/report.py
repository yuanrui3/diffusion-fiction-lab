"""Turn scored rows into the report tables (Markdown).

Usage:
    python scripts/report.py results/pilot_scored.jsonl --out results/pilot_report.md
    python scripts/report.py results/full_scored.jsonl --out results/full_tables.md

Tables:
  1. Per model, set, condition and direction: pass rate per rule, all-rules pass
     rate with a 95% interval, parallelism, relevance, exact match, distinct
     answers per 8.
  2. Gap between the unseen and recall sets, per model, condition and direction
     (only when both sets are present).
  3. Each diffusion model minus Qwen on the same items, and how that difference
     changes between the unseen and recall sets.
  4. The reference couplets scored by the same checker (the ceiling).

Intervals are bootstrap intervals over items (see dfl/stats.py).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

from dfl.checker import check_reference
from dfl.scoring import default_tagger
from dfl.stats import by_item, diff_independent, diff_of_diffs, diff_paired, mean_ci
from dfl.tones import Pingshui

ROOT = Path(__file__).resolve().parents[1]
ITEM_FILES = {"unseen": "data/unseen/huohou.jsonl", "recall": "data/recall/recall.jsonl", "dev": "data/dev/dev.jsonl"}
MODELS = ["qwen", "dream", "llada", "dream-default", "llada-default"]
RULES = [
    ("rule_length", "Length"),
    ("rule_tone_modern", "End tone (modern)"),
    ("rule_tone_classical", "End tone (classical)"),
    ("rule_no_repeat", "No repeat"),
]
GAP_METRICS = [
    ("all_rules_modern", "All rules"), ("rule_length", "Length"), ("rule_tone_modern", "End tone"),
    ("rule_no_repeat", "No repeat"), ("pos_match_all", "Parallelism"), ("relevance", "Relevance"),
    ("exact", "Exact match"),
]


def pct(x) -> str:
    return "–" if x is None else f"{100 * x:.0f}%"


def pct_ci(r: dict) -> str:
    if r["mean"] is None:
        return "–"
    return f"{100 * r['mean']:.0f}% [{100 * r['lo']:.0f}, {100 * r['hi']:.0f}]"


def num(x, digits=2) -> str:
    return "–" if x is None else f"{x:.{digits}f}"


def pts(r: dict, scale: float = 100.0, unit: str = "") -> str:
    if r.get("diff") is None:
        return "–"
    star = " *" if (r["lo"] > 0 or r["hi"] < 0) else ""
    return f"{scale * r['diff']:+.0f}{unit} [{scale * r['lo']:+.0f}, {scale * r['hi']:+.0f}]{star}"


def pts2(r: dict) -> str:
    if r.get("diff") is None:
        return "–"
    star = " *" if (r["lo"] > 0 or r["hi"] < 0) else ""
    return f"{r['diff']:+.2f} [{r['lo']:+.2f}, {r['hi']:+.2f}]{star}"


def fmt_gap(key: str, r: dict) -> str:
    return pts2(r) if key in ("relevance", "pos_match_all") else pts(r)


def table(header: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(out)


def load_items() -> dict[str, dict]:
    items = {}
    for path in ITEM_FILES.values():
        p = ROOT / path
        if p.exists():
            for line in p.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    it = json.loads(line)
                    items[it["id"]] = it
    return items


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("scored", nargs="+", type=Path)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--no-ceiling", action="store_true")
    args = ap.parse_args()

    rows = []
    for path in args.scored:
        rows += [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    for r in rows:
        r["pos_match_all"] = r["pos_match"] or 0.0
    items = load_items()

    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        groups[(r["model"], r["set"], r["condition"], r["direction"])].append(r)
    sets = [s for s in ("unseen", "recall", "dev") if any(k[1] == s for k in groups)]
    conditions = [c for c in ("summary", "bare") if any(k[2] == c for k in groups)]
    directions = ["forward", "reverse"]
    models = [m for m in MODELS if any(k[0] == m for k in groups)]

    out: list[str] = []

    # ---- 1. main table
    out.append("## 1. Pass rates and scores\n")
    out.append("Per sample. “All rules” = length, modern end tone and no repeat together, and not a copy of the given line; "
               "its bracket is a 95% bootstrap interval over items. Parallelism counts a wrong-length line as 0. "
               "Relevance is cosine similarity to the summary (bge-m3).\n")
    header = ["Set", "Condition", "Direction", "Model", "Items", *[label for _, label in RULES],
              "All rules", "All rules (classical tone)", "Parallelism", "Relevance", "Exact match", "Distinct per 8"]
    body = []
    for s in sets:
        for c in conditions:
            for d in directions:
                for m in models:
                    g = groups.get((m, s, c, d))
                    if not g:
                        continue
                    per_item = defaultdict(list)
                    for r in g:
                        per_item[r["item_id"]].append(r["answer"])
                    rate = lambda key: sum(bool(r[key]) for r in g) / len(g)
                    rel = [r["relevance"] for r in g if r["relevance"] is not None]
                    body.append([
                        s, c, d, m, str(len(per_item)),
                        *[pct(rate(key)) for key, _ in RULES],
                        pct_ci(mean_ci(by_item(g, "all_rules_modern"))),
                        pct(rate("all_rules_classical")),
                        num(sum(r["pos_match_all"] for r in g) / len(g)),
                        num(sum(rel) / len(rel)) if rel else "–",
                        pct(rate("exact")),
                        num(sum(len(set(v)) for v in per_item.values()) / len(per_item), 1),
                    ])
    out.append(table(header, body))

    # ---- 2. unseen minus recall
    if "unseen" in sets and "recall" in sets:
        out.append("\n## 2. Unseen minus recall\n")
        out.append("Same model, same condition and direction; percentage points (score points for parallelism and relevance), "
                   "95% interval, * = interval excludes zero. “Fully covered” drops the recall chapters whose text does not "
                   "contain one of the two events.\n")
        header = ["Condition", "Direction", "Model", "Recall items", *[label for _, label in GAP_METRICS]]
        body = []
        for c in conditions:
            for d in directions:
                for m in models:
                    u = groups.get((m, "unseen", c, d))
                    rc = groups.get((m, "recall", c, d))
                    if not u or not rc:
                        continue
                    variants = [("all", rc)]
                    if c == "summary":
                        full = [r for r in rc if items.get(r["item_id"], {}).get("summary_covers", "both") == "both"]
                        if 0 < len(full) < len(rc):
                            variants.append(("fully covered", full))
                    for name, sub in variants:
                        body.append([c, d, m, f"{len({r['item_id'] for r in sub})} ({name})",
                                     *[fmt_gap(key, diff_independent(by_item(u, key), by_item(sub, key))) for key, _ in GAP_METRICS]])
        out.append(table(header, body))

    # ---- 3. diffusion minus qwen
    if "qwen" in models and len(models) > 1:
        out.append("\n## 3. Diffusion model minus Qwen, on the same items\n")
        out.append("Percentage points (score points for parallelism and relevance), 95% interval, * = interval excludes zero.\n")
        header = ["Set", "Condition", "Direction", "Model", *[label for _, label in GAP_METRICS]]
        body = []
        for s in sets:
            for c in conditions:
                for d in directions:
                    q = groups.get(("qwen", s, c, d))
                    for m in models:
                        g = groups.get((m, s, c, d))
                        if m == "qwen" or not g or not q:
                            continue
                        body.append([s, c, d, f"{m} − qwen",
                                     *[fmt_gap(key, diff_paired(by_item(g, key), by_item(q, key))) for key, _ in GAP_METRICS]])
        out.append(table(header, body))

        if "unseen" in sets and "recall" in sets:
            out.append("\n### Does that difference change between the sets?\n")
            out.append("(diffusion − Qwen on unseen) minus (diffusion − Qwen on recall). Positive means the diffusion model "
                       "does relatively better on the unseen set.\n")
            header = ["Condition", "Direction", "Model", *[label for _, label in GAP_METRICS]]
            body = []
            for c in conditions:
                for d in directions:
                    for m in models:
                        if m == "qwen":
                            continue
                        need = [groups.get((m, "unseen", c, d)), groups.get(("qwen", "unseen", c, d)),
                                groups.get((m, "recall", c, d)), groups.get(("qwen", "recall", c, d))]
                        if not all(need):
                            continue
                        body.append([c, d, f"{m} − qwen",
                                     *[fmt_gap(key, diff_of_diffs(*[by_item(g, key) for g in need])) for key, _ in GAP_METRICS]])
            out.append(table(header, body))

    # ---- 4. ceilings
    if not args.no_ceiling:
        pingshui = Pingshui.load()
        tagger, coarse = default_tagger()
        out.append("\n## 4. The original couplets under the same checker\n")
        out.append("Each original line scored as if it had been generated from the other. This is the ceiling for the rules.\n")
        header = ["Set", "Direction", "Items", *[label for _, label in RULES], "All rules", "All rules (classical tone)", "Parallelism"]
        body = []
        used = {r["item_id"] for r in rows}
        for s in sets:
            its = [it for it in items.values() if it.get("set") == s and it["id"] in used]
            if not its:
                continue
            res = [check_reference(it["first"], it["second"], pingshui=pingshui, tagger=tagger, coarse=coarse) for it in its]
            for d in directions:
                g = [r[d] for r in res]
                rate = lambda key: sum(bool(r[key]) for r in g) / len(g)
                body.append([s, d, str(len(g)), *[pct(rate(key)) for key, _ in RULES], pct(rate("all_rules_modern")),
                             pct(rate("all_rules_classical")), num(sum((r["pos_match"] or 0.0) for r in g) / len(g))])
        out.append(table(header, body))

    text = "\n".join(out) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8", newline="\n")
        print(f"wrote {args.out}")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
