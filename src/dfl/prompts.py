"""The task prompt. One user message, the same text for every model.

The prompt states every rule the checker applies, so the test is whether a
model can meet known constraints, not whether it happens to know the
conventions of 回目. It never asserts anything about the tone of the *given*
line, because some original lines break the convention.
"""

from __future__ import annotations

from typing import Optional

PROMPT_VERSION = "v2"   # chosen on dev (PROTOCOL.md, entry 27b); v1 is kept for the record

FORWARD = "forward"   # given the first line (上句), write the second (下句)
REVERSE = "reverse"   # given the second line (下句), write the first (上句)

_NUMERALS = "零一二三四五六七八九十"


def chinese_numeral(n: int) -> str:
    """1–99 as a Chinese numeral (七, 十, 十二, 二十, 二十一)."""
    if not 0 < n < 100:
        raise ValueError(f"only 1–99 supported, got {n}")
    tens, ones = divmod(n, 10)
    if tens == 0:
        return _NUMERALS[ones]
    head = "十" if tens == 1 else _NUMERALS[tens] + "十"
    return head + (_NUMERALS[ones] if ones else "")


_ROLE = {
    FORWARD: {"given": "上句", "want": "下句", "verb": "请对出下句。",
              "tone": "下句须以平声字收尾（回目的惯例是上句收仄、下句收平）"},
    REVERSE: {"given": "下句", "want": "上句", "verb": "请补出上句。",
              "tone": "上句须以仄声字收尾（回目的惯例是上句收仄、下句收平）"},
}


VERSIONS = ("v1", "v2")


def build_prompt(
    given: str, direction: str, n_chars: int, summary: Optional[str] = None, version: str = PROMPT_VERSION
) -> str:
    """The user message for one item.

    `n_chars` is the required length of the line to be written, in 汉字.
    With a `summary` the model is also asked to fit it.

    Two layouts with the same wording:
      v1  task sentence, material (summary and given line), rules, closing instruction
      v2  task sentence, rules, closing instruction, material, and last of all the
          empty label of the line to be written (“下句：”), so the reply fills a slot
    """
    if direction not in _ROLE:
        raise ValueError(f"direction must be {FORWARD!r} or {REVERSE!r}, got {direction!r}")
    if version not in VERSIONS:
        raise ValueError(f"version must be one of {VERSIONS}, got {version!r}")
    r = _ROLE[direction]
    n = chinese_numeral(n_chars)

    if summary:
        intro = f"下面是一部章回体小说某一回的内容梗概，以及这一回回目的{r['given']}。{r['verb']}"
        material = f"梗概：{summary}\n{r['given']}：{given}"
    else:
        intro = f"下面是一部章回体小说某一回回目的{r['given']}。{r['verb']}"
        material = f"{r['given']}：{given}"
    head = f"{intro}\n\n{material}"

    rules = [
        f"{r['want']}必须正好{n}个字，与{r['given']}字数相同",
        f"与{r['given']}对仗，词性、节奏逐字相对",
        r["tone"],
        f"不要重复{r['given']}用过的字",
    ]
    if summary:
        rules.append("内容要切合梗概")
    body = "\n".join(f"{_NUMERALS[i]}、{rule}{'；' if i < len(rules) else '。'}" for i, rule in enumerate(rules, 1))

    tail = f"只写出{r['want']}本身，不加标点，不作解释。"
    if version == "v1":
        return f"{head}\n\n要求：\n{body}\n\n{tail}"
    return f"{intro}\n\n要求：\n{body}\n\n{tail}\n\n{material}\n{r['want']}："


def wanted_line(direction: str) -> str:
    """Which line the model writes: "second" for forward, "first" for reverse."""
    return {FORWARD: "second", REVERSE: "first"}[direction]
