"""Turn a model's raw reply into the one line it was asked for.

The same rule is applied to every model, autoregressive or diffusion:

1. Take the non-empty lines of the reply. Drop lines that start with the label
   of the line that was *given* (上句/上联 when the second line was asked for,
   and the reverse), unless that would drop everything.
2. If a remaining line starts with the label of the *requested* line, start
   from the first such line. Otherwise start from the first remaining line.
3. In that line, if there is a colon, keep only what follows the last colon
   ("下句：……", "好的，下句是：……"). Strip whitespace, quotation marks,
   brackets and punctuation from both ends.
4. If nothing is left (the line was only a label or a lead-in), do step 3 on
   the next line instead.
5. What remains is the answer. Nothing inside it is removed.

The answer is "pure" if every remaining character is a 汉字. The checker's
length rule needs a pure answer of exactly the required length, so a reply
that buries the line in commentary on the same line fails honestly.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .text import count_hanzi, is_pure_hanzi

# Label patterns for each line. "second" = 下句/下联, "first" = 上句/上联.
_LABELS = {
    "second": re.compile(r"^\s*(?:回目)?(?:的)?下[句联聯]"),
    "first": re.compile(r"^\s*(?:回目)?(?:的)?上[句联聯]"),
}
_OTHER = {"second": "first", "first": "second"}

_COLON_RE = re.compile(r"[:：]")

# Characters stripped from both ends of the candidate line.
_EDGE_CHARS = (
    " \t\r\n　 "
    "\"'`“”‘’「」『』《》〈〉【】〔〕（）()[]{}<>"
    "，。、；：！？,.;:!?…—－-~～·*#"
)


@dataclass
class Extraction:
    answer: str                 # the extracted line (may be empty)
    pure: bool                  # every character of `answer` is a 汉字
    n_hanzi: int                # number of 汉字 in `answer`
    n_lines: int                # non-empty lines in the raw reply
    notes: list[str] = field(default_factory=list)


def _candidate(line: str) -> str:
    """Step 3: what follows the last colon, with the edges stripped."""
    return _COLON_RE.split(line)[-1].strip(_EDGE_CHARS)


def extract_answer(raw: str, want: str) -> Extraction:
    """Extract the requested line from a raw reply.

    `want` is "second" (forward direction: the model writes the second line) or
    "first" (reverse direction: the model writes the first line).
    """
    if want not in _LABELS:
        raise ValueError(f"want must be 'first' or 'second', got {want!r}")

    all_lines = [ln for ln in (raw or "").splitlines() if ln.strip()]
    notes: list[str] = []
    if not all_lines:
        return Extraction("", False, 0, 0, ["empty reply"])
    if len(all_lines) > 1:
        notes.append("reply had more than one line")

    given_label = _LABELS[_OTHER[want]]
    lines = [ln for ln in all_lines if not given_label.match(ln)]
    if len(lines) < len(all_lines):
        notes.append("skipped a line labelled as the given line")
    if not lines:
        lines = all_lines

    wanted_label = _LABELS[want]
    start = next((i for i, ln in enumerate(lines) if wanted_label.match(ln)), 0)
    if start > 0:
        notes.append("picked the line by its label")

    answer = _candidate(lines[start])
    if not answer and start + 1 < len(lines):
        answer = _candidate(lines[start + 1])
        notes.append("label line was empty; took the next line")
    if not answer:
        notes.append("nothing left after stripping")

    return Extraction(
        answer=answer,
        pure=is_pure_hanzi(answer),
        n_hanzi=count_hanzi(answer),
        n_lines=len(all_lines),
        notes=notes,
    )
