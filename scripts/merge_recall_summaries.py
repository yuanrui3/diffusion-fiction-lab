"""Check the recall summaries against the rules and merge them into the recall set.

Usage:
    python scripts/merge_recall_summaries.py            # check only
    python scripts/merge_recall_summaries.py --write    # check, then write

Reads the summarisers' raw output in `data/recall/summaries_raw/`, applies the
review edits listed below, and checks every summary the same way as the unseen
set: 2–3 sentences, 60–110 汉字, no run of three or more characters shared with
the 回目 other than the names listed in ALLOWED.

With --write (and only if every check passes) it writes
`data/recall/recall_summaries.json` and fills the `summary` fields of the 48
sampled items in `data/recall/recall.jsonl`.

`covers` records what the summarisers found when the chapter text does not
contain the event one line of the 回目 names (in these novels a chapter often
breaks off before the second event, which opens the next chapter):
    both              both events are in the chapter text and in the summary
    one_line_weak     one event is only begun or only partly there
    one_line_missing  one event is not in the chapter text at all
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

from dfl.text import count_hanzi

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "recall" / "summaries_raw"
RECALL = ROOT / "data" / "recall" / "recall.jsonl"
SAMPLE = ROOT / "data" / "recall" / "summary_sample.json"
OUT = ROOT / "data" / "recall" / "recall_summaries.json"

# Names of people and places (and book titles) that a summary may share with its 回目.
ALLOWED: set[str] = {
    "蒋玉菡", "大观园", "莺脰湖", "匡超人", "明月岭", "萧云仙", "青枫城", "野羊塘", "元武阁",
    "博望坡", "诸葛亮", "葭萌关", "野猪林", "安平寨", "飞云浦", "鲁智深", "高老庄", "流沙河", "豹头山",
}

# Wording edits made in review: id -> list of (from, to).
EDITS: dict[str, list[tuple[str, str]]] = {}

COVERS = {
    "shz-008": "one_line_missing",
    "shz-028": "one_line_missing",
    "rlws-018": "one_line_missing",
    "hshlz-056": "one_line_missing",
    "shz-029": "one_line_weak",
    "rlws-012": "one_line_weak",
}


def shared_runs(summary: str, line: str) -> set[str]:
    """Maximal runs of 3+ characters that occur in both."""
    hits = set()
    for size in range(len(line), 2, -1):
        for i in range(len(line) - size + 1):
            run = line[i:i + size]
            if run in summary and not any(run in h for h in hits):
                hits.add(run)
    return hits


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    write = "--write" in sys.argv
    items = [json.loads(line) for line in RECALL.read_text(encoding="utf-8").splitlines() if line.strip()]
    by_id = {it["id"]: it for it in items}
    sample = json.loads(SAMPLE.read_text(encoding="utf-8"))["ids"]

    raw: dict[str, dict] = {}
    for path in sorted(RAW.glob("*.json")):
        for rec in json.loads(path.read_text(encoding="utf-8")):
            raw[rec["id"]] = rec

    problems = []
    merged = {}
    for item_id in sample:
        if item_id not in raw:
            problems.append(f"{item_id}: no summary")
            continue
        it = by_id[item_id]
        summary = raw[item_id]["summary"]
        edited = False
        for old, new in EDITS.get(item_id, []):
            if old not in summary:
                problems.append(f"{item_id}: edit target not found: {old}")
            summary = summary.replace(old, new)
            edited = True
        n = count_hanzi(summary)
        sentences = len(re.findall(r"[。！？]", summary))
        runs = (shared_runs(summary, it["first"]) | shared_runs(summary, it["second"]))
        bad = sorted(r for r in runs if r not in ALLOWED and not any(r in a for a in ALLOWED))
        flags = []
        if not 60 <= n <= 110:
            flags.append(f"{n} 汉字")
        if not 2 <= sentences <= 3:
            flags.append(f"{sentences} sentences")
        if bad:
            flags.append(f"shares {bad}")
        status = "; ".join(flags) if flags else "ok"
        print(f"{item_id}  {it['first']}　{it['second']}  [{n} 字, {sentences} 句]  {status}")
        if flags:
            problems.append(f"{item_id}: {status}")
        merged[item_id] = {
            "summary": summary,
            "summary_by": "Claude",
            "summary_edited_in_review": edited,
            "summary_covers": COVERS.get(item_id, "both"),
            "summariser_note": raw[item_id].get("note", ""),
        }
    extra = sorted(set(raw) - set(sample))
    if extra:
        problems.append(f"summaries for items outside the sample: {extra}")

    print(f"\n{len(merged)} of {len(sample)} summaries; {len(problems)} problems")
    if problems:
        print("\n".join(problems))
        return 1
    if not write:
        print("all checks pass (run with --write to merge)")
        return 0

    OUT.write_text(json.dumps(merged, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n")
    with open(RECALL, "w", encoding="utf-8", newline="\n") as fh:
        for it in items:
            if it["id"] in merged:
                it.update(merged[it["id"]])
            fh.write(json.dumps(it, ensure_ascii=False) + "\n")
    print(f"wrote {OUT.name} and filled {len(merged)} items in {RECALL.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
