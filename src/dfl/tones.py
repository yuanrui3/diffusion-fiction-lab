"""平/仄 classes for single characters: modern Mandarin and classical (平水韵).

Modern tones come from pypinyin, read in the context of the whole line so that
common 多音字 resolve sensibly. Tones 1 and 2 are 平, 3 and 4 are 仄. A neutral
tone falls back to the character's citation tone.

Classical tones come from a table built from the Wikisource text of 《平水韻》
(`scripts/build_pingshui.py`). 平 is 平; 上, 去 and 入 are 仄. A character
listed under both a 平 rhyme and a 仄 rhyme is BOTH and satisfies either
requirement. That makes the classical check lenient for characters such as 看
or 中, whose tone depends on the sense; the table cannot tell the senses apart.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Optional, Sequence

PING = "ping"
ZE = "ze"
BOTH = "both"
UNKNOWN = "unknown"

DEFAULT_PINGSHUI_PATH = Path(__file__).resolve().parents[2] / "data" / "lexicon" / "pingshui.json"


# --------------------------------------------------------------------------
# Modern Mandarin
# --------------------------------------------------------------------------

def tone_number(reading: str) -> int:
    """'zhong1' -> 1, 'de5' -> 5, 'de' -> 5 (no digit means neutral)."""
    if reading and reading[-1].isdigit():
        return int(reading[-1])
    return 5


def class_of_tone(tone: int) -> str:
    if tone in (1, 2):
        return PING
    if tone in (3, 4):
        return ZE
    return UNKNOWN


def _pypinyin_line(line: str) -> list[str]:
    """One TONE3 reading per character of `line`, read in context."""
    from pypinyin import Style, pinyin

    out = pinyin(line, style=Style.TONE3, heteronym=False, neutral_tone_with_five=True)
    readings = [item[0] for item in out]
    if len(readings) != len(line):
        # A character pypinyin does not know can be merged with its neighbours.
        # Fall back to reading the characters one at a time to keep alignment.
        readings = []
        for ch in line:
            one = pinyin(ch, style=Style.TONE3, heteronym=False, neutral_tone_with_five=True)
            readings.append(one[0][0] if one and one[0] else ch)
    return readings


def _pypinyin_char(ch: str) -> list[str]:
    """All TONE3 readings pypinyin lists for one character."""
    from pypinyin import Style, pinyin

    out = pinyin(ch, style=Style.TONE3, heteronym=True, neutral_tone_with_five=True)
    return list(out[0]) if out else []


LineReader = Callable[[str], Sequence[str]]
CharReader = Callable[[str], Sequence[str]]


def modern_tones(
    line: str,
    *,
    line_reader: LineReader = _pypinyin_line,
    char_reader: CharReader = _pypinyin_char,
) -> list[int]:
    """Tone number (1–4, or 5 if no toned reading exists) for each character.

    The readers are injectable so that tests do not depend on pypinyin's data.
    """
    tones: list[int] = []
    for ch, reading in zip(line, line_reader(line)):
        tone = tone_number(reading) if _looks_like_pinyin(reading) else 5
        if tone == 5:
            toned = [tone_number(r) for r in char_reader(ch) if _looks_like_pinyin(r)]
            toned = [t for t in toned if t != 5]
            tone = toned[0] if toned else 5
        tones.append(tone)
    return tones


def _looks_like_pinyin(reading: str) -> bool:
    return bool(reading) and reading[0].isascii() and reading[0].isalpha()


def modern_classes(line: str, **readers) -> list[str]:
    return [class_of_tone(t) for t in modern_tones(line, **readers)]


def modern_is_ambiguous(ch: str, *, char_reader: CharReader = _pypinyin_char) -> bool:
    """True if the character has both a 平 and a 仄 reading in modern Mandarin."""
    classes = {class_of_tone(tone_number(r)) for r in char_reader(ch) if _looks_like_pinyin(r)}
    return PING in classes and ZE in classes


# --------------------------------------------------------------------------
# Classical (平水韵)
# --------------------------------------------------------------------------

class Pingshui:
    """Lookup of classical tone categories, keyed by character.

    `table` maps a character to a string made of the letters 平, 上, 去, 入
    (one or more).
    """

    def __init__(self, table: dict[str, str], meta: Optional[dict] = None):
        self.table = table
        self.meta = meta or {}

    @classmethod
    def load(cls, path: Path | str = DEFAULT_PINGSHUI_PATH) -> "Pingshui":
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        return cls(data["chars"], data.get("meta"))

    def tones(self, ch: str) -> str:
        return self.table.get(ch, "")

    def klass(self, ch: str) -> str:
        tones = self.tones(ch)
        if not tones:
            return UNKNOWN
        has_ping = "平" in tones
        has_ze = any(t in tones for t in "上去入")
        if has_ping and has_ze:
            return BOTH
        return PING if has_ping else ZE

    def classes(self, line: str) -> list[str]:
        return [self.klass(ch) for ch in line]

    def is_entering(self, ch: str) -> bool:
        """True if the character has an 入声 reading."""
        return "入" in self.tones(ch)


def satisfies(klass: str, need: str) -> Optional[bool]:
    """Does a character of class `klass` meet the requirement `need` (PING or ZE)?

    Returns None when the class is unknown, so the caller can fall back.
    """
    if klass == UNKNOWN:
        return None
    return klass == BOTH or klass == need
