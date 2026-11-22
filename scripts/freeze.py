"""Record or verify the SHA-256 of every frozen file.

Usage:
    python scripts/freeze.py          # verify FROZEN.sha256
    python scripts/freeze.py --write  # (re)write it from the list below

The list is everything a result depends on and that must not change after the
freeze: the three sets, the tone table, the frozen settings, the prompt, the
extraction rule, the checker, the scoring code and the generation code.
`--write` is for the freeze itself; afterwards only verification is legitimate,
and any change to a listed file needs a dated entry in PROTOCOL.md.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "FROZEN.sha256"

FROZEN = [
    "data/unseen/huohou.jsonl",
    "data/recall/recall.jsonl",
    "data/recall/recall_summaries.json",
    "data/recall/summary_sample.json",
    "data/recall/sources.json",
    "data/dev/dev.jsonl",
    "data/dev/splits.json",
    "data/lexicon/pingshui.json",
    "config/phase1.json",
    "src/dfl/prompts.py",
    "src/dfl/extract.py",
    "src/dfl/text.py",
    "src/dfl/tones.py",
    "src/dfl/parallel.py",
    "src/dfl/checker.py",
    "src/dfl/scoring.py",
    "src/dfl/stats.py",
    "src/dfl/generate.py",
    "src/dfl/models.py",
    "src/dfl/registry.py",
    "scripts/run_generation.py",
    "scripts/run_plan.py",
    "scripts/score.py",
    "scripts/report.py",
    "scripts/make_grading_sheet.py",
    "scripts/unblind.py",
    "PLAN_FROZEN.md",
    "uv.lock",
]


def digest(path: Path) -> str:
    # Hash with line endings normalised, so a checkout on another system verifies too.
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def main() -> int:
    if "--write" in sys.argv:
        missing = [f for f in FROZEN if not (ROOT / f).exists()]
        if missing:
            print("cannot freeze, missing:", ", ".join(missing))
            return 1
        MANIFEST.write_text("".join(f"{digest(ROOT / f)}  {f}\n" for f in FROZEN), encoding="utf-8", newline="\n")
        print(f"wrote {MANIFEST.name} with {len(FROZEN)} files")
        return 0

    bad = 0
    for line in MANIFEST.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        want, name = line.split(None, 1)
        path = ROOT / name.strip()
        if not path.exists():
            print(f"MISSING  {name}")
            bad += 1
        elif digest(path) != want:
            print(f"CHANGED  {name}")
            bad += 1
        else:
            print(f"ok       {name}")
    print("all frozen files verify" if not bad else f"{bad} file(s) differ from the freeze")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
