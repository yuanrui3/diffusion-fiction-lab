"""Score one or more run files with the checker and print the summary table.

Usage:
    python scripts/score.py runs/smoke/*.jsonl --show 4
    python scripts/score.py runs/pilot/*.jsonl --relevance --out results/pilot_scored.jsonl

--show N      also print the first N samples of every record, with their flags
--relevance   embed answers and summaries with bge-m3 (needs the GPU free, or runs on CPU)
--items F     item files to take summaries from (default: all three sets)
--out F       write one scored row per sample as JSONL
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

from dfl.scoring import add_relevance, aggregate, default_tagger, format_table, score_record
from dfl.tones import Pingshui

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ITEMS = ["data/unseen/huohou.jsonl", "data/recall/recall.jsonl", "data/dev/dev.jsonl"]

COLUMNS = [
    "model", "set", "condition", "direction", "n_items", "n_samples",
    "rule_length", "rule_tone_modern", "rule_tone_classical", "rule_no_repeat",
    "all_rules_modern", "all_rules_classical", "pos_match", "even_opposition_modern",
    "relevance", "exact", "echo", "distinct_per_item",
]


def load_summaries(paths: list[str]) -> dict[str, str]:
    out = {}
    for p in paths:
        path = ROOT / p
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                it = json.loads(line)
                if it.get("summary"):
                    out[it["id"]] = it["summary"]
    return out


def flags(row: dict) -> str:
    marks = [
        "L" if row["rule_length"] else "l",
        "T" if row["rule_tone_modern"] else "t",
        "C" if row["rule_tone_classical"] else "c",
        "R" if row["rule_no_repeat"] else "r",
    ]
    extra = []
    if row["exact"]:
        extra.append("=ref")
    if row["echo"]:
        extra.append("echo")
    if row["junk_after_end"]:
        extra.append("junk")
    if row["pos_match"] is not None:
        extra.append(f"pos {row['pos_match']:.2f}")
    if row["relevance"] is not None:
        extra.append(f"rel {row['relevance']:.2f}")
    return "".join(marks) + (" " + " ".join(extra) if extra else "")


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--show", type=int, default=0)
    ap.add_argument("--relevance", action="store_true")
    ap.add_argument("--items", nargs="*", default=DEFAULT_ITEMS)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    files = sorted({f for pattern in args.runs for f in glob.glob(pattern)})
    if not files:
        print("no run files matched")
        return 1

    pingshui = Pingshui.load()
    tagger, coarse = default_tagger()
    records = []
    for f in files:
        for line in Path(f).read_text(encoding="utf-8").splitlines():
            rec = json.loads(line)
            if rec.get("type") == "item":
                records.append(rec)

    rows = []
    per_record = []
    for rec in records:
        scored = score_record(rec, pingshui=pingshui, tagger=tagger, coarse=coarse)
        per_record.append((rec, scored))
        rows.extend(scored)

    if args.relevance:
        from sentence_transformers import SentenceTransformer
        import torch

        from dfl.registry import EMBEDDER, local_path

        device = "cuda" if torch.cuda.is_available() else "cpu"
        embedder = SentenceTransformer(str(local_path(EMBEDDER)), device=device)
        add_relevance(rows, load_summaries(args.items), embedder)

    if args.show:
        for rec, scored in per_record:
            print(f"\n[{rec['model']}] {rec['item_id']} {rec['direction']} {rec['condition']}  "
                  f"given: {rec['given']}  reference: {rec['reference']}  "
                  f"({rec['seconds']} s, {rec['n_forward']} forward passes)")
            for row in scored[: args.show]:
                raw = row["raw"].replace("\n", "⏎")
                shown = row["answer"] if row["answer"] == raw.strip() else f"{row['answer']}   <- raw: {raw[:40]}"
                print(f"    {flags(row):28s} {shown}")
        print("\nflags: L length, T modern end tone, C classical end tone, R no repeat; lower case = failed")

    print()
    print(format_table(aggregate(rows, ["model", "set", "condition", "direction"]), COLUMNS))

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", encoding="utf-8", newline="\n") as fh:
            for row in rows:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"\nwrote {len(rows)} rows to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
