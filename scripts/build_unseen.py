"""Check the frozen unseen set against the novel's public 洁本 files.

Usage:
    python scripts/build_unseen.py [path to the Slow-fire-novel clone]

The unseen set (`data/unseen/huohou.jsonl`) is frozen: its summaries were
written once and are not regenerated. This script re-reads the chapter
headings and 回目 from the source clone and verifies that every item's
chapter, title and two lines match the source, that no chapter with a 回目 is
missing, and that every summary still obeys the wording rule (no run of three
or more characters shared with its 回目, other than the listed names).

It never rewrites the frozen file. A mismatch is reported and the exit code is 1.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

from dfl.text import count_hanzi

ROOT = Path(__file__).resolve().parents[1]
FROZEN = ROOT / "data" / "unseen" / "huohou.jsonl"
DEFAULT_SOURCE = ROOT.parent / "_sources" / "Slow-fire-novel"

# Personal names that may be shared between a summary and its 回目.
ALLOWED_NAMES = {"伊内丝", "萨凡纳", "莉迪娅", "阿丝玛", "蒲嬢嬢", "以斯拉", "布雷登", "徐好翔", "路易斯", "优素福"}

HEADING = re.compile(r"^#\s*(\S+?)　+(.+)$")


def read_source(source: Path) -> dict[int, dict]:
    out = {}
    for path in sorted((source / "火候" / "chapters").glob("*.md")):
        number = int(path.name[:2])
        lines = [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
        m = HEADING.match(lines[0])
        if not m or lines[1].startswith("**"):      # no 回目 under the title
            continue
        parts = re.split(r"　+", lines[1])
        if len(parts) != 2:
            raise SystemExit(f"{path.name}: cannot split the 回目 line: {lines[1]}")
        out[number] = {"chapter_label": m.group(1), "title": m.group(2), "first": parts[0], "second": parts[1]}
    return out


def shared_runs(summary: str, line: str) -> set[str]:
    hits = {line[i:i + 3] for i in range(len(line) - 2) if line[i:i + 3] in summary}
    return {h for h in hits if h not in ALLOWED_NAMES}


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    source = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_SOURCE
    src = read_source(source)
    items = [json.loads(line) for line in FROZEN.read_text(encoding="utf-8").splitlines() if line.strip()]

    problems = []
    seen = set()
    for it in items:
        n = it["chapter"]
        seen.add(n)
        s = src.get(n)
        if s is None:
            problems.append(f"{it['id']}: chapter {n} has no 回目 in the source")
            continue
        for key in ("chapter_label", "title", "first", "second"):
            if it[key] != s[key]:
                problems.append(f"{it['id']}: {key} differs: frozen {it[key]!r}, source {s[key]!r}")
        runs = shared_runs(it["summary"], it["first"]) | shared_runs(it["summary"], it["second"])
        if runs:
            problems.append(f"{it['id']}: summary shares {sorted(runs)} with the 回目")
        if not 60 <= count_hanzi(it["summary"]) <= 110:
            problems.append(f"{it['id']}: summary has {count_hanzi(it['summary'])} 汉字")
    for n in sorted(set(src) - seen):
        problems.append(f"chapter {n} has a 回目 in the source but no item")

    print(f"source: {source}")
    print(f"frozen items: {len(items)}, source chapters with a 回目: {len(src)}")
    if problems:
        print("\n".join(problems))
        return 1
    print("unseen set matches the source; all summaries obey the wording rule")
    return 0


if __name__ == "__main__":
    sys.exit(main())
