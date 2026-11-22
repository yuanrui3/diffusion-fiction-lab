"""Build the recall set and the dev pool from Wikisource contents pages.

Usage:
    python scripts/build_recall.py

Reads the 回目 of six public-domain 章回 novels (the recall set) and of 《鏡花緣》
(the dev pool) from the contents pages of Chinese Wikisource, converts them to
simplified characters with OpenCC, and writes

    data/recall/recall.jsonl     one couplet per line
    data/dev/dev_pool.jsonl      one couplet per line
    data/recall/sources.json     page titles and the revision each was read at

On the first run the current revision of each page is used and recorded. Later
runs read exactly the recorded revisions, so the sets can be rebuilt unchanged.

Parsing rules:
- A contents line looks like `*[[/第001回|第一回]]　上句　下句` or, for 紅樓夢,
  `*[[/第001回|第一回　上句　下句]]`.
- `{{另|A|B}}` (a variant reading) and `{{參|A|note}}` are replaced by A, and
  the no-conversion markup `-{A}-` by A.
- Conversion to simplified characters is OpenCC `t2s`, then a short list of
  variant fixes (see VARIANT_FIXES) so the lines read as common simplified
  editions print them.
- The two lines are whatever is separated by whitespace after the chapter label.
  A contents line that does not yield exactly two parts made only of 汉字 is
  skipped and reported.
"""

from __future__ import annotations

import json
import re
import sys
import time
import urllib.parse
import urllib.request
from collections import Counter
from pathlib import Path

from opencc import OpenCC

from dfl.text import is_pure_hanzi

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
SOURCES = ROOT / "data" / "recall" / "sources.json"
API = "https://zh.wikisource.org/w/api.php"
USER_AGENT = "couplet-research-script/0.1 (local research)"

RECALL = [
    {"key": "hlm", "work": "红楼梦", "page": "紅樓夢", "expect": 120},
    {"key": "shz", "work": "水浒传", "page": "水滸傳 (120回本)", "expect": 120},
    {"key": "sgyy", "work": "三国演义", "page": "三國演義", "expect": 120},
    {"key": "xyj", "work": "西游记", "page": "西遊記", "expect": 100},
    {"key": "rlws", "work": "儒林外史", "page": "儒林外史", "expect": 56},
    {"key": "hshlz", "work": "海上花列传", "page": "海上花列傳", "expect": 64},
]
DEV = [{"key": "jhy", "work": "镜花缘", "page": "鏡花緣", "expect": 100}]

LINE_RE = re.compile(r"^\*+\s*\[\[/第\s*0*(\d+)\s*回\|([^\]]*)\]\](.*)$")
LABEL_RE = re.compile(r"^\s*第[一二三四五六七八九十百零〇○两兩]+回")
VARIANT_RE = re.compile(r"\{\{[另參参]\|([^|{}]*)\|[^{}]*\}\}")
NOCONVERT_RE = re.compile(r"-\{([^{}]*)\}-")
TEMPLATE_RE = re.compile(r"\{\{[^{}]*\}\}")
TAG_RE = re.compile(r"<[^>]+>")

# Traditional characters whose simplified form depends on the sense. Lines that
# contain one are listed after the build for a check by eye.
AMBIGUOUS = set("著乾瞭於餘徵麼髮發鍾鐘裏裡喫祇隻雲閒鬥鬆闢颳夥嚮繫鑒鑑複復覆後后並幷臺檯颱係系")


# Variant characters that OpenCC leaves alone but common simplified editions
# print in the standard form. Applied after conversion.
VARIANT_FIXES = {"餽": "馈", "擡": "抬", "渰": "淹", "姪": "侄", "雠": "仇",
                 "慇": "殷", "懃": "勤", "倣": "仿", "珮": "佩"}
PHRASE_FIXES = {"祇怕": "只怕"}


def to_simplified(trad: str, cc: OpenCC) -> str:
    """Convert one line. A character whose simplified form lies outside the
    Basic Multilingual Plane (a formal but rarely printed simplification, such
    as 綝 to U+2C62D) is kept as it stands in the source."""
    simp = cc.convert(trad)
    if len(simp) != len(trad):
        raise ValueError(f"conversion changed the length: {trad} -> {simp}")
    simp = "".join(t if ord(s) > 0xFFFF else s for s, t in zip(simp, trad))
    simp = "".join(VARIANT_FIXES.get(ch, ch) for ch in simp)
    for a, b in PHRASE_FIXES.items():
        simp = simp.replace(a, b)
    return simp


def fetch(page: str, revid: int | None) -> tuple[str, int]:
    params = {"action": "parse", "prop": "wikitext|revid", "format": "json", "formatversion": 2}
    if revid:
        params["oldid"] = revid
    else:
        params.update(page=page, redirects=1)
    req = urllib.request.Request(f"{API}?{urllib.parse.urlencode(params)}", headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = json.load(resp)["parse"]
    return data["wikitext"], data["revid"]


def parse_contents(wikitext: str) -> tuple[dict[int, tuple[str, str]], list[str]]:
    couplets: dict[int, tuple[str, str]] = {}
    skipped: list[str] = []
    for raw in wikitext.splitlines():
        m = LINE_RE.match(raw.strip())
        if not m:
            continue
        chapter = int(m.group(1))
        text = f"{m.group(2)} {m.group(3)}"
        text = NOCONVERT_RE.sub(lambda v: v.group(1), text)
        text = VARIANT_RE.sub(lambda v: v.group(1), text)
        text = TEMPLATE_RE.sub(" ", text)
        text = TAG_RE.sub(" ", text)
        text = LABEL_RE.sub(" ", text, count=1)
        parts = [p for p in re.split(r"[\s　]+", text) if p]
        if len(parts) != 2 or not all(is_pure_hanzi(p) for p in parts):
            skipped.append(raw.strip())
            continue
        if chapter in couplets:
            skipped.append(f"duplicate chapter {chapter}: {raw.strip()}")
            continue
        couplets[chapter] = (parts[0], parts[1])
    return couplets, skipped


def build(works: list[dict], set_name: str, sources: dict, cc: OpenCC) -> list[dict]:
    items: list[dict] = []
    for w in works:
        known = sources.get(w["page"], {}).get("revid")
        wikitext, revid = fetch(w["page"], known)
        sources[w["page"]] = {"revid": revid, "key": w["key"], "work": w["work"]}
        RAW.mkdir(parents=True, exist_ok=True)
        (RAW / f"contents_{w['key']}.txt").write_text(wikitext, encoding="utf-8")
        couplets, skipped = parse_contents(wikitext)
        print(f"{w['work']} ({w['page']}, rev {revid}): {len(couplets)} couplets, expected {w['expect']}, skipped {len(skipped)}")
        for s in skipped:
            print(f"    skipped: {s[:100]}")
        for chapter in sorted(couplets):
            t1, t2 = couplets[chapter]
            s1, s2 = to_simplified(t1, cc), to_simplified(t2, cc)
            items.append({
                "id": f"{w['key']}-{chapter:03d}",
                "set": set_name,
                "work": w["work"],
                "work_key": w["key"],
                "chapter": chapter,
                "first": s1,
                "second": s2,
                "first_trad": t1,
                "second_trad": t2,
                "summary": None,
            })
        time.sleep(1.0)   # be polite to the API
    return items


def write_jsonl(path: Path, items: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        for it in items:
            fh.write(json.dumps(it, ensure_ascii=False) + "\n")


def report(items: list[dict], label: str) -> None:
    lengths = Counter((len(it["first"]), len(it["second"])) for it in items)
    unequal = [it for it in items if len(it["first"]) != len(it["second"])]
    print(f"\n{label}: {len(items)} items")
    print("  line lengths (first, second): " + ", ".join(f"{k}: {v}" for k, v in sorted(lengths.items())))
    print(f"  unequal-length couplets: {len(unequal)}" + "".join(f"\n    {it['id']} {it['first']}　{it['second']}" for it in unequal))
    def in_gb2312(ch: str) -> bool:
        try:
            ch.encode("gb2312")
            return True
        except UnicodeEncodeError:
            return False

    rare = [(it, [c for c in it["first"] + it["second"] if not in_gb2312(c)]) for it in items]
    rare = [(it, cs) for it, cs in rare if cs]
    print(f"  lines with a character outside GB2312 (rare or variant): {len(rare)}")
    for it, cs in rare:
        print(f"    {it['id']}  {it['first']}　{it['second']}  [{' '.join(cs)}]")
    flagged = [it for it in items if AMBIGUOUS & set(it["first_trad"] + it["second_trad"])]
    print(f"  lines with a character that converts ambiguously: {len(flagged)}")
    for it in flagged:
        print(f"    {it['id']}  {it['first_trad']}　{it['second_trad']}  ->  {it['first']}　{it['second']}")


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    sources = json.loads(SOURCES.read_text(encoding="utf-8")) if SOURCES.exists() else {}
    cc = OpenCC("t2s")
    recall = build(RECALL, "recall", sources, cc)
    merged = ROOT / "data" / "recall" / "recall_summaries.json"
    if merged.exists():      # summaries merged by scripts/merge_recall_summaries.py survive a rebuild
        extra = json.loads(merged.read_text(encoding="utf-8"))
        for it in recall:
            if it["id"] in extra:
                it.update(extra[it["id"]])
    dev = build(DEV, "dev", sources, cc)
    write_jsonl(ROOT / "data" / "recall" / "recall.jsonl", recall)
    write_jsonl(ROOT / "data" / "dev" / "dev_pool.jsonl", dev)
    SOURCES.parent.mkdir(parents=True, exist_ok=True)
    SOURCES.write_text(json.dumps(sources, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n")
    report(recall, "recall set")
    report(dev, "dev pool")
    return 0


if __name__ == "__main__":
    sys.exit(main())
