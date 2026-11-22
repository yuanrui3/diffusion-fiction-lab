"""Build the classical tone table from the Wikisource text of 《平水韻》.

Usage:
    python scripts/build_pingshui.py

Fetches one pinned revision of https://zh.wikisource.org/wiki/平水韻 (public
domain text, marked {{PD-old}}), reads the five tone sections, and writes
`data/lexicon/pingshui.json`:

    {"meta": {...}, "chars": {"东": "平", "看": "平去", "白": "入", ...}}

Each character maps to the tone categories it is listed under, as a string of
the letters 平 上 去 入. Keys include the traditional characters of the source
and their simplified forms (OpenCC, character by character), so a simplified
character inherits the tones of every traditional character that simplifies to
it. Lines marked 【詞】 on the page (characters added for 词 rhyming) are included
and counted separately in the metadata.
"""

from __future__ import annotations

import json
import re
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

from opencc import OpenCC

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "lexicon" / "pingshui.json"
RAW = ROOT / "data" / "raw" / "pingshui_wikitext.txt"

PAGE = "平水韻"
REVID = 8264804
API = "https://zh.wikisource.org/w/api.php"
USER_AGENT = "couplet-research-script/0.1 (local research)"

SECTIONS = {"上平聲部": "平", "下平聲部": "平", "上聲部": "上", "去聲部": "去", "入聲部": "入"}
ORDER = "平上去入"
HAN = re.compile(r"[㐀-䶿一-鿿豈-﫿\U00020000-\U0003134f]")


def fetch_wikitext() -> str:
    query = urllib.parse.urlencode(
        {"action": "parse", "oldid": REVID, "prop": "wikitext", "format": "json", "formatversion": 2}
    )
    req = urllib.request.Request(f"{API}?{query}", headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = json.load(resp)
    return data["parse"]["wikitext"]


def parse(wikitext: str) -> tuple[dict[str, set[str]], dict]:
    """Return {traditional char: set of tone letters} and counts."""
    table: dict[str, set[str]] = {}
    counts = {"rhyme_groups": 0, "main": 0, "ci": 0}
    tone = None
    for line in wikitext.splitlines():
        line = line.strip()
        if not line:
            continue
        m = re.match(r"^==\s*(.+?)\s*==$", line)
        if m:
            tone = SECTIONS.get(m.group(1))
            continue
        if tone is None or line.startswith("{{") or line.startswith("[["):
            continue
        if "{{++|" in line:          # a rhyme-group heading such as 上平聲一{{++|東}}
            counts["rhyme_groups"] += 1
            continue
        is_ci = line.startswith("【詞】")
        chars = HAN.findall(line[3:] if is_ci else line)
        counts["ci" if is_ci else "main"] += len(chars)
        for ch in chars:
            table.setdefault(ch, set()).add(tone)
    return table, counts


def main() -> int:
    wikitext = fetch_wikitext()
    RAW.parent.mkdir(parents=True, exist_ok=True)
    RAW.write_text(wikitext, encoding="utf-8")

    trad, counts = parse(wikitext)
    if counts["rhyme_groups"] != 106:
        print(f"expected 106 rhyme groups, found {counts['rhyme_groups']}; not writing", file=sys.stderr)
        return 1

    cc = OpenCC("t2s")
    merged: dict[str, set[str]] = {}
    for ch, tones in trad.items():
        merged.setdefault(ch, set()).update(tones)
        simp = cc.convert(ch)
        if len(simp) == 1 and simp != ch:
            merged.setdefault(simp, set()).update(tones)

    chars = {ch: "".join(t for t in ORDER if t in tones) for ch, tones in sorted(merged.items())}
    both = sum(1 for t in chars.values() if "平" in t and any(x in t for x in "上去入"))
    meta = {
        "source": f"https://zh.wikisource.org/wiki/{urllib.parse.quote(PAGE)}",
        "page": PAGE,
        "revid": REVID,
        "licence": "public-domain text ({{PD-old}}); Wikisource site content CC BY-SA 4.0",
        "built": time.strftime("%Y-%m-%d"),
        "rhyme_groups": counts["rhyme_groups"],
        "source_characters_main": counts["main"],
        "source_characters_ci": counts["ci"],
        "distinct_traditional": len(trad),
        "keys_total": len(chars),
        "keys_ping_and_ze": both,
        "keys_with_entering_tone": sum(1 for t in chars.values() if "入" in t),
        "note": "Simplified keys inherit the tones of every traditional character that simplifies to them.",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", encoding="utf-8", newline="\n") as fh:
        json.dump({"meta": meta, "chars": chars}, fh, ensure_ascii=False, indent=0, sort_keys=False)
        fh.write("\n")
    print(json.dumps(meta, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
