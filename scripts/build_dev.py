"""Build the dev set: the 《镜花缘》 pool with the few dev summaries merged in.

Usage:
    python scripts/build_dev.py

Writes `data/dev/dev.jsonl`. It also names the items used for the smoke test
(3) and the pilot (10), so those choices are fixed in one place:

- smoke: three items that have a dev summary.
- pilot: the eight items that have a dev summary, plus two drawn from the rest
  of the pool with a fixed seed.
- calibration: twenty further items, none of them in the pilot, used to pick the
  diffusion sampling temperature.

Before writing, it checks the dev summaries against the same wording rule as
the test summaries (no run of three or more characters shared with the 回目).
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
POOL = ROOT / "data" / "dev" / "dev_pool.jsonl"
SUMMARIES = ROOT / "data" / "dev" / "dev_summaries.json"
OUT = ROOT / "data" / "dev" / "dev.jsonl"
SPLITS = ROOT / "data" / "dev" / "splits.json"

SMOKE = ["jhy-003", "jhy-007", "jhy-033"]
PILOT_SEED = 20261119


def shared_runs(summary: str, line: str, n: int = 3) -> list[str]:
    return sorted({line[i:i + n] for i in range(len(line) - n + 1) if line[i:i + n] in summary})


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    pool = [json.loads(line) for line in POOL.read_text(encoding="utf-8").splitlines() if line.strip()]
    summaries = {k: v for k, v in json.loads(SUMMARIES.read_text(encoding="utf-8")).items() if not k.startswith("_")}
    by_id = {it["id"]: it for it in pool}

    problems = []
    for item_id, summary in summaries.items():
        it = by_id[item_id]
        runs = shared_runs(summary, it["first"]) + shared_runs(summary, it["second"])
        if runs:
            problems.append(f"{item_id}: shares {runs} with its 回目")
        it["summary"] = summary
        it["summary_by"] = "Claude, from memory (dev only, unchecked)"
    if problems:
        print("\n".join(problems))
        return 1

    with_summary = sorted(summaries)
    rest = [it["id"] for it in pool if it["id"] not in summaries]
    extra = random.Random(PILOT_SEED).sample(rest, 2)
    pilot = with_summary + sorted(extra)
    outside = [i for i in rest if i not in pilot]
    calibration = sorted(random.Random(PILOT_SEED + 1).sample(outside, 20))
    splits = {"smoke": SMOKE, "pilot": pilot, "calibration": calibration, "pilot_seed": PILOT_SEED}

    with open(OUT, "w", encoding="utf-8", newline="\n") as fh:
        for it in pool:
            fh.write(json.dumps(it, ensure_ascii=False) + "\n")
    SPLITS.write_text(json.dumps(splits, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(f"dev set: {len(pool)} items, {len(summaries)} with a summary")
    print(f"smoke: {splits['smoke']}")
    print(f"pilot: {splits['pilot']}")
    print(f"calibration: {splits['calibration']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
