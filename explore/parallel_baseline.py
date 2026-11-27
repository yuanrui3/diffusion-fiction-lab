"""How much of the parallelism score is about the given line? A shuffled baseline.

Usage:
    python explore/parallel_baseline.py        # LTP on the CPU; writes explore/out/parallel_baseline.md

Post hoc and exploratory (PROTOCOL.md entry 36). For a seeded sample of
answers that have the right length, the frozen parallelism score against the
line actually given (taken from the scored rows) is compared with the same
score against a different item's given line of the same length, role and set.
If a model aligned its words to the given line position by position, the
first should clearly exceed the second; if it only wrote a generic 回目-shaped
line, the two will be close. The original couplets are scored the same way.
"""

from __future__ import annotations

import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dfl.parallel import ltp_coarse, make_ltp_tagger, parallelism  # noqa: E402
from dfl.registry import TAGGER, local_path  # noqa: E402
from dfl.text import hanzi_only  # noqa: E402

OUT = ROOT / "explore" / "out"
MODELS = ["qwen", "llada", "dream", "llada-default", "dream-default"]
NAMES = {"qwen": "Qwen", "llada": "LLaDA 0.7", "dream": "Dream 0.5", "llada-default": "LLaDA 0", "dream-default": "Dream 0.2"}
PER_CELL = 800
SEED = 20261125


def boot_diff(pairs_by_item, n_boot=2000, seed=SEED):
    ids = sorted(pairs_by_item)
    obs = np.array([np.mean([a for a, _ in pairs_by_item[i]]) for i in ids])
    base = np.array([np.mean([b for _, b in pairs_by_item[i]]) for i in ids])
    w = np.array([len(pairs_by_item[i]) for i in ids], dtype=float)
    rng = np.random.default_rng(seed)
    d = []
    for _ in range(n_boot):
        k = rng.integers(0, len(ids), len(ids))
        d.append(np.average(obs[k], weights=w[k]) - np.average(base[k], weights=w[k]))
    est = np.average(obs, weights=w) - np.average(base, weights=w)
    return np.average(obs, weights=w), np.average(base, weights=w), est, np.percentile(d, 2.5), np.percentile(d, 97.5)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    tagger = make_ltp_tagger(str(local_path(TAGGER)), device="cpu")
    rng = random.Random(SEED)

    items = {}
    for f in ["data/unseen/huohou.jsonl", "data/recall/recall.jsonl"]:
        for line in open(ROOT / f, encoding="utf-8"):
            it = json.loads(line)
            items[it["id"]] = it
    # given lines by (set, role, length): role is the given line's role
    pool = defaultdict(list)
    for it in items.values():
        for role, line in (("first", it["first"]), ("second", it["second"])):
            pool[(it["set"], role, len(hanzi_only(line)))].append((it["id"], line))

    rows = defaultdict(list)
    with open(ROOT / "results" / "full_scored.jsonl", encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            if r["model"] in MODELS and r.get("pos_match") is not None:
                rows[(r["model"], r["set"])].append(r)

    report = ["# Parallelism against a shuffled baseline\n",
              "Post hoc; see `PROTOCOL.md` entry 36. *Own*: the frozen parallelism score (LTP base2, share of positions with the same word class) "
              "against the line actually given. *Shuffled*: the same answer against another item's given line of the same length and role, from the same set. "
              f"Seeded sample of up to {PER_CELL} right-length answers per model and set; 95% bootstrap intervals over items.\n",
              "| Writer | Set | Answers | Own | Shuffled | Own − shuffled |", "|---|---|---|---|---|---|"]
    data = {}
    for set_name in ("unseen", "recall"):
        # the originals
        by_item = defaultdict(list)
        for it in items.values():
            if it["set"] != set_name:
                continue
            for given_role, given, written in (("first", it["first"], it["second"]), ("second", it["second"], it["first"])):
                cands = [g for i, g in pool[(set_name, given_role, len(hanzi_only(given)))] if i != it["id"]]
                if not cands or len(hanzi_only(written)) != len(hanzi_only(given)):
                    continue
                own = parallelism(hanzi_only(given), hanzi_only(written), tagger=tagger, coarse=ltp_coarse)["pos_match"]
                other = parallelism(hanzi_only(rng.choice(cands)), hanzi_only(written), tagger=tagger, coarse=ltp_coarse)["pos_match"]
                if own is not None and other is not None:
                    by_item[it["id"]].append((own, other))
        o, b, d, lo, hi = boot_diff(by_item)
        data[f"originals|{set_name}"] = [o, b, d, lo, hi]
        report.append(f"| Originals | {set_name} | {sum(len(v) for v in by_item.values())} | {o:.2f} | {b:.2f} | {d:+.2f} [{lo:+.2f}, {hi:+.2f}] |")
        for m in MODELS:
            cand_rows = rows[(m, set_name)]
            sample = rng.sample(cand_rows, min(PER_CELL, len(cand_rows)))
            by_item = defaultdict(list)
            for r in sample:
                given_role = "first" if r["direction"] == "forward" else "second"
                cands = [g for i, g in pool[(set_name, given_role, len(hanzi_only(r["given"])))] if i != r["item_id"]]
                if not cands:
                    continue
                # own score recomputed with the same CPU tagger, so both columns come from one tagger run
                # hanzi only, as the frozen checker does before scoring parallelism
                ans = hanzi_only(r["answer"] or "")
                own = parallelism(hanzi_only(r["given"]), ans, tagger=tagger, coarse=ltp_coarse)["pos_match"]
                other = parallelism(hanzi_only(rng.choice(cands)), ans, tagger=tagger, coarse=ltp_coarse)["pos_match"]
                if own is not None and other is not None:
                    by_item[r["item_id"]].append((own, other))
            o, b, d, lo, hi = boot_diff(by_item)
            data[f"{m}|{set_name}"] = [o, b, d, lo, hi]
            report.append(f"| {NAMES[m]} | {set_name} | {sum(len(v) for v in by_item.values())} | {o:.2f} | {b:.2f} | {d:+.2f} [{lo:+.2f}, {hi:+.2f}] |")
            print(report[-1], flush=True)
    (OUT / "parallel_baseline.md").write_text("\n".join(report) + "\n", encoding="utf-8", newline="\n")
    (OUT / "parallel_baseline.json").write_text(json.dumps(data, indent=1, default=float) + "\n", encoding="utf-8", newline="\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
