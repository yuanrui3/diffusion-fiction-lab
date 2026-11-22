"""Unblind the author's grades and summarise them.

Usage:
    python scripts/unblind.py [results/grading]

Needs `grades.json` (exported from the grading sheet) beside the key. Refuses to
run without it, so the key is never read before grading is done. Writes
`graded.jsonl` (one graded line per row, with its model) and `grades_report.md`.

Each of the 17 cells (chapter, direction) was graded once per model, so the
model comparison is paired by cell; intervals are bootstrap intervals over cells.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

from dfl.stats import diff_paired, mean_ci

ROOT = Path(__file__).resolve().parents[1]
MODELS = ["qwen", "dream", "llada"]


def fmt(r: dict, key: str = "mean") -> str:
    if r.get(key) is None:
        return "–"
    return f"{r[key]:.2f} [{r['lo']:.2f}, {r['hi']:.2f}]"


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    folder = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "results" / "grading"
    grades_path = folder / "grades.json"
    if not grades_path.exists():
        print(f"{grades_path} not found: grading is not finished, the key stays closed")
        return 1
    grades = json.loads(grades_path.read_text(encoding="utf-8"))
    key = json.loads((folder / "KEY_DO_NOT_OPEN.json").read_text(encoding="utf-8"))
    if grades.get("sheet") != key.get("sheet"):
        print(f"grades are for sheet {grades.get('sheet')!r}, the key is for {key.get('sheet')!r}")
        return 1

    rows = []
    missing = []
    for g in grades["grades"]:
        meta = key["key"][g["entry"]]
        if g["score"] is None:
            missing.append(g["entry"])
        rows.append({**meta, "entry": g["entry"], "score": g["score"], "comment": g.get("comment", "")})
    with open(folder / "graded.jsonl", "w", encoding="utf-8", newline="\n") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    graded = [r for r in rows if r["score"] is not None]
    by_model = {m: defaultdict(list) for m in MODELS}
    for r in graded:
        by_model[r["model"]][f"{r['item_id']}|{r['direction']}"].append(r["score"])

    out = ["# Human grades, unblinded\n"]
    out.append(f"{len(graded)} of {len(rows)} lines graded" + (f"; not graded: {', '.join(missing)}" if missing else "") + ".\n")
    out.append("Literary quality, 1 to 5, by the author. One line per model for each of the same chapters and directions. "
               "Brackets are 95% bootstrap intervals over those chapter-direction cells.\n")
    out.append("| Model | Lines | Mean score | Lines scored 4 or 5 | Drawn from rule-passing samples |")
    out.append("|---|---|---|---|---|")
    for m in MODELS:
        mine = [r for r in graded if r["model"] == m]
        if not mine:
            continue
        good = sum(r["score"] >= 4 for r in mine)
        passed = sum(r["passed_all_rules"] for r in mine)
        out.append(f"| {m} | {len(mine)} | {fmt(mean_ci(by_model[m]))} | {good} | {passed} |")

    out.append("\n## Differences on the same cells\n")
    out.append("| Comparison | Cells | Difference in mean score |")
    out.append("|---|---|---|")
    for a, b in (("dream", "qwen"), ("llada", "qwen"), ("llada", "dream")):
        d = diff_paired(by_model[a], by_model[b])
        if d.get("diff") is not None:
            star = " *" if (d["lo"] > 0 or d["hi"] < 0) else ""
            out.append(f"| {a} − {b} | {d['n_items']} | {d['diff']:+.2f} [{d['lo']:+.2f}, {d['hi']:+.2f}]{star} |")

    out.append("\n## By direction\n")
    out.append("| Model | Forward (writes the second line) | Reverse (writes the first line) |")
    out.append("|---|---|---|")
    for m in MODELS:
        cells = []
        for d in ("forward", "reverse"):
            sub = {k: v for k, v in by_model[m].items() if k.endswith("|" + d)}
            cells.append(fmt(mean_ci(sub)) if sub else "–")
        out.append(f"| {m} | {cells[0]} | {cells[1]} |")

    out.append("\n## Every graded line\n")
    out.append("| Entry | Chapter | Direction | Model | Line | Passed all rules | Score | Comment |")
    out.append("|---|---|---|---|---|---|---|---|")
    for r in sorted(rows, key=lambda r: (r["item_id"], r["direction"], r["model"])):
        comment = (r["comment"] or "").replace("|", "/").replace("\n", " ")
        out.append(f"| {r['entry']} | {r['item_id']} | {r['direction']} | {r['model']} | {r['answer']} | "
                   f"{'yes' if r['passed_all_rules'] else 'no'} | {r['score'] if r['score'] is not None else '–'} | {comment} |")

    (folder / "grades_report.md").write_text("\n".join(out) + "\n", encoding="utf-8", newline="\n")
    print("\n".join(out[:20]))
    print(f"\nwrote {folder / 'graded.jsonl'} and {folder / 'grades_report.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
