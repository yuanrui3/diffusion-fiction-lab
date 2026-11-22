"""Which tagger makes the parallelism score separate real couplets from mismatched pairs?

Usage:
    python scripts/compare_taggers.py [tagger names]
    python scripts/compare_taggers.py --download      # fetch the LTP and Stanza models first

For every reference couplet in the recall and dev sets, the true pair
(first line, its own second line) is scored against a mismatched pair (the same
first line with the second line of another couplet of the same length, drawn
with a fixed seed). A useful score puts the true pair higher. No model output
is involved, and the unseen set is shown but not used to choose.

Taggers compared: jieba.posseg (words), jieba's dictionary applied character by
character, LTP small, Stanza zh-hans (modern), Stanza lzh (classical Chinese).
"""

from __future__ import annotations

import json
import random
import sys
import time
from pathlib import Path

from dfl.parallel import (
    jieba_char_tagger, jieba_coarse, jieba_tagger, ltp_coarse, make_ltp_tagger,
    make_stanza_tagger, parallelism, upos_coarse,
)
from dfl.registry import models_dir

ROOT = Path(__file__).resolve().parents[1]
SETS = {
    "recall": "data/recall/recall.jsonl",
    "dev": "data/dev/dev.jsonl",
    "unseen (report only)": "data/unseen/huohou.jsonl",
}
SEED = 20261119
ALL = ["jieba", "jieba-char", "ltp-small", "stanza-zh", "stanza-lzh"]


def download() -> None:
    import urllib.request

    import stanza
    from huggingface_hub import snapshot_download

    m = models_dir()
    info = json.load(urllib.request.urlopen("https://huggingface.co/api/models/LTP/small", timeout=60))
    licence = [t for t in info.get("tags", []) if t.startswith("license:")]
    print(f"LTP/small commit {info['sha']} licence tag {licence}")
    snapshot_download(repo_id="LTP/small", revision=info["sha"], local_dir=m / "ltp-small")
    for lang in ("zh-hans", "lzh"):
        stanza.download(lang, model_dir=str(m / "stanza"), processors="tokenize,pos", verbose=False)
        print(f"stanza {lang} downloaded")


def pairs(items: list[dict], rng: random.Random) -> list[tuple[str, str, str]]:
    by_len: dict[int, list[str]] = {}
    for it in items:
        by_len.setdefault(len(it["second"]), []).append(it["second"])
    out = []
    for it in items:
        pool = [s for s in by_len.get(len(it["first"]), []) if s != it["second"]]
        if pool and len(it["first"]) == len(it["second"]):
            out.append((it["first"], it["second"], rng.choice(pool)))
    return out


def mean(rows: list[dict], key: str) -> float:
    return sum(r[key] for r in rows) / len(rows)


def evaluate(name: str, tagger, coarse, data: dict) -> None:
    for set_name, triples in data.items():
        t0 = time.time()
        true = [parallelism(a, b, tagger=tagger, coarse=coarse) for a, b, _ in triples]
        fake = [parallelism(a, c, tagger=tagger, coarse=coarse) for a, _, c in triples]
        ok = [(t, f) for t, f in zip(true, fake) if t["pos_match"] is not None and f["pos_match"] is not None]
        n = len(ok)
        wins = sum(t["pos_match"] > f["pos_match"] for t, f in ok)
        ties = sum(t["pos_match"] == f["pos_match"] for t, f in ok)
        both = [(t["pos_match"] + t["boundary_match"], f["pos_match"] + f["boundary_match"]) for t, f in ok]
        wins2 = sum(a > b for a, b in both)
        ties2 = sum(a == b for a, b in both)
        print(
            f"{name:11s} {set_name:21s} n={n:3d}/{len(triples):3d}  "
            f"pos {mean([t for t, _ in ok], 'pos_match'):.3f} vs {mean([f for _, f in ok], 'pos_match'):.3f}  "
            f"win {wins / n:.0%} tie {ties / n:.0%} -> {(wins + ties / 2) / n:.0%}  |  "
            f"boundary {mean([t for t, _ in ok], 'boundary_match'):.3f} vs {mean([f for _, f in ok], 'boundary_match'):.3f}  |  "
            f"pos+boundary -> {(wins2 + ties2 / 2) / n:.0%}  ({time.time() - t0:.0f} s)",
            flush=True,
        )


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    args = sys.argv[1:]
    if "--download" in args:
        download()
        args.remove("--download")

    rng = random.Random(SEED)
    data = {}
    for set_name, path in SETS.items():
        items = [json.loads(line) for line in (ROOT / path).read_text(encoding="utf-8").splitlines() if line.strip()]
        data[set_name] = pairs(items, rng)

    print("columns: true pair vs mismatched pair; win = true pair scores higher; arrow = win + half the ties\n")
    m = models_dir()
    for name in args or ALL:
        if name == "jieba":
            evaluate(name, jieba_tagger, jieba_coarse, data)
        elif name == "jieba-char":
            evaluate(name, jieba_char_tagger, jieba_coarse, data)
        elif name == "ltp-small":
            evaluate(name, make_ltp_tagger(str(m / "ltp-small"), device="cuda"), ltp_coarse, data)
        elif name == "stanza-zh":
            evaluate(name, make_stanza_tagger("zh-hans", str(m / "stanza"), use_gpu=True), upos_coarse, data)
        elif name == "stanza-lzh":
            evaluate(name, make_stanza_tagger("lzh", str(m / "stanza"), use_gpu=True), upos_coarse, data)
        else:
            print(f"unknown tagger {name}")
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
