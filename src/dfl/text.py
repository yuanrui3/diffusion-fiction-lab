"""Small helpers for counting and picking out 汉字."""

from __future__ import annotations

import re

# CJK Unified Ideographs, Extension A, the compatibility block, the supplementary
# planes, and 〇 (U+3007), which Chinese numerals such as 二〇四八 use as a character.
_HANZI_CLASS = (
    "〇"
    "㐀-䶿"
    "一-鿿"
    "豈-﫿"
    "\U00020000-\U0003134f"
)
HANZI_RE = re.compile(f"[{_HANZI_CLASS}]")
NON_HANZI_RE = re.compile(f"[^{_HANZI_CLASS}]")


def is_hanzi(ch: str) -> bool:
    """True if the single character `ch` is a 汉字."""
    return len(ch) == 1 and HANZI_RE.fullmatch(ch) is not None


def hanzi_only(s: str) -> str:
    """`s` with everything that is not a 汉字 removed."""
    return NON_HANZI_RE.sub("", s)


def count_hanzi(s: str) -> int:
    return len(HANZI_RE.findall(s))


def is_pure_hanzi(s: str) -> bool:
    """True if `s` is non-empty and every character is a 汉字."""
    return bool(s) and NON_HANZI_RE.search(s) is None
