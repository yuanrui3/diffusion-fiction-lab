"""Pick the recall chapters that get a summary, and fetch their text.

Usage:
    python scripts/fetch_recall_chapters.py

Draws 8 chapters per novel from the recall set with a fixed seed (48 in all),
fetches each chapter page from Chinese Wikisource, strips the wiki markup,
converts the text to simplified characters, and writes

    data/recall/summary_sample.json        the chosen item ids, the seed, and the page revisions read
    data/raw/recall_chapters/<id>.txt      the chapter text, 回目 on the first line (not in git)

The chapter texts are working material for the summarisers only. They stay in
`data/raw/`, which git ignores.
"""

from __future__ import annotations

import json
import random
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from opencc import OpenCC

ROOT = Path(__file__).resolve().parents[1]
RECALL = ROOT / "data" / "recall" / "recall.jsonl"
SOURCES = ROOT / "data" / "recall" / "sources.json"
SAMPLE = ROOT / "data" / "recall" / "summary_sample.json"
RAW = ROOT / "data" / "raw"
OUT = RAW / "recall_chapters"
API = "https://zh.wikisource.org/w/api.php"
USER_AGENT = "couplet-research-script/0.1 (local research)"

SEED = 20261119
PER_WORK = 8

LINK_RE = re.compile(r"^\*+\s*\[\[/(第\s*0*(\d+)\s*回)\|")


def subpages(key: str) -> dict[int, str]:
    """Chapter number -> subpage name, from the saved contents page."""
    out = {}
    for line in (RAW / f"contents_{key}.txt").read_text(encoding="utf-8").splitlines():
        m = LINK_RE.match(line.strip())
        if m:
            out.setdefault(int(m.group(2)), m.group(1))
    return out


def fetch(title: str, revid: int | None) -> tuple[str, int]:
    params = {"action": "parse", "prop": "wikitext|revid", "format": "json", "formatversion": 2}
    if revid:
        params["oldid"] = revid
    else:
        params.update(page=title, redirects=1)
    req = urllib.request.Request(f"{API}?{urllib.parse.urlencode(params)}", headers={"User-Agent": USER_AGENT})
    for attempt in range(6):
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = json.load(resp)
            break
        except urllib.error.HTTPError as err:
            if err.code != 429 or attempt == 5:
                raise
            wait = int(err.headers.get("Retry-After") or 0) or 20 * (attempt + 1)
            print(f"    rate limited, waiting {wait} s", flush=True)
            time.sleep(wait)
    if "parse" not in data:
        raise RuntimeError(f"{title}: {data.get('error', data)}")
    return data["parse"]["wikitext"], data["parse"]["revid"]


def clean(wikitext: str) -> str:
    """Wiki markup to plain text. Rough on purpose: the summariser reads prose."""
    t = wikitext
    t = re.sub(r"<!--.*?-->", "", t, flags=re.S)
    t = re.sub(r"<ref[^>]*?/>|<ref[^>]*>.*?</ref>", "", t, flags=re.S)
    t = re.sub(r"-\{([^{}]*)\}-", r"\1", t)
    t = re.sub(r"\{\{[另參参]\|([^|{}]*)\|[^{}]*\}\}", r"\1", t)
    for _ in range(6):                                  # nested templates, innermost first
        t, n = re.subn(r"\{\{[^{}]*\}\}", "", t)
        if not n:
            break
    t = re.sub(r"\[\[(?:File|Image|Category|文件|分類|分类)[^\]]*\]\]", "", t, flags=re.I)
    t = re.sub(r"\[\[[^\]|]*\|([^\]]*)\]\]", r"\1", t)
    t = re.sub(r"\[\[([^\]]*)\]\]", r"\1", t)
    t = re.sub(r"<[^>]+>", "", t)
    t = t.replace("'''", "").replace("''", "")
    t = re.sub(r"^=+\s*(.*?)\s*=+$", r"\1", t, flags=re.M)
    t = re.sub(r"^[:;*#]+\s*", "", t, flags=re.M)
    t = re.sub(r"[ \t　]+\n", "\n", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    items = [json.loads(line) for line in RECALL.read_text(encoding="utf-8").splitlines() if line.strip()]
    sources = json.loads(SOURCES.read_text(encoding="utf-8"))
    page_of = {v["key"]: page for page, v in sources.items()}
    previous = json.loads(SAMPLE.read_text(encoding="utf-8")) if SAMPLE.exists() else {}
    known_revs = previous.get("chapter_revisions", {})

    by_work: dict[str, list[dict]] = {}
    for it in items:
        by_work.setdefault(it["work_key"], []).append(it)

    rng = random.Random(SEED)
    chosen: list[dict] = []
    for key in sorted(by_work):                         # fixed order, so the draw is reproducible
        chosen.extend(sorted(rng.sample(by_work[key], PER_WORK), key=lambda it: it["chapter"]))

    cc = OpenCC("t2s")
    OUT.mkdir(parents=True, exist_ok=True)
    revisions = {}
    for it in chosen:
        sub = subpages(it["work_key"])[it["chapter"]]
        title = f"{page_of[it['work_key']]}/{sub}"
        wikitext, revid = fetch(title, known_revs.get(it["id"]))
        revisions[it["id"]] = revid
        text = cc.convert(clean(wikitext))
        body = f"回目：{it['first']}　{it['second']}\n\n{text}\n"
        (OUT / f"{it['id']}.txt").write_text(body, encoding="utf-8", newline="\n")
        n_han = len(re.findall(r"[一-鿿]", text))
        flag = "  <-- SHORT, check" if n_han < 2500 else ""
        print(f"{it['id']}  {it['work']} 第{it['chapter']}回  {it['first']}　{it['second']}  {n_han} 汉字{flag}")
        time.sleep(0.5)

    SAMPLE.write_text(
        json.dumps({"seed": SEED, "per_work": PER_WORK, "ids": [it["id"] for it in chosen],
                    "chapter_revisions": revisions}, ensure_ascii=False, indent=1) + "\n",
        encoding="utf-8", newline="\n",
    )
    print(f"\n{len(chosen)} chapters chosen; texts in {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
