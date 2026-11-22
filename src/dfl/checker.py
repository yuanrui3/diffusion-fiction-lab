"""The couplet checker.

Two entry points:

- `check_generated(given, answer, direction, ...)` scores one generated line
  against the line the model was given.
- `check_reference(first, second, ...)` scores an original couplet the same way,
  once per direction, as the ceiling.

Rules (pass or fail), applied to the line being judged:

1. length      The line is made only of 汉字 and has as many as the other line.
2. end tone    A first line ends 仄; a second line ends 平. Reported with modern
               Mandarin tones and with classical (平水韵) tones.
3. no repeat   No character occurs in both lines, unless it sits at the same
               positions in both.

`all_rules` is 1 and 2 and 3, and also requires that the line is not a plain
copy of the other line: a copy would pass rule 3 through the same-position
exception, and it is not a couplet line.

Scores (not part of `all_rules`):

- even-position tone opposition: share of positions 2, 4, 6… at which the two
  lines carry opposite tones;
- word-class parallelism and word-boundary agreement (see `parallel.py`).

Rules 2 and 3 are evaluated on the 汉字 of the line, so a line that fails rule 1
because of stray punctuation is still told whether its tone and characters
were right. An empty line fails everything.
"""

from __future__ import annotations

from typing import Optional

from .parallel import Tagger, jieba_coarse, jieba_tagger, parallelism
from .text import hanzi_only, is_pure_hanzi
from .tones import BOTH, PING, UNKNOWN, ZE, Pingshui, class_of_tone, modern_tones, satisfies

FORWARD = "forward"   # given the first line, write the second
REVERSE = "reverse"   # given the second line, write the first

# What the last character of each line must be.
END_NEED = {"first": ZE, "second": PING}


def repeated_characters(first: str, second: str) -> list[str]:
    """Characters that break rule 3, in order of first appearance in `first`.

    A character shared by both lines is allowed only if the set of positions it
    occupies is the same in both lines.
    """
    bad: list[str] = []
    for ch in dict.fromkeys(first):
        if ch not in second:
            continue
        pos1 = {i for i, c in enumerate(first) if c == ch}
        pos2 = {i for i, c in enumerate(second) if c == ch}
        if pos1 != pos2:
            bad.append(ch)
    return bad


def same_position_count(first: str, second: str) -> int:
    """Number of positions at which both lines have the same character."""
    return sum(a == b for a, b in zip(first, second))


def _end_info(line: str, pingshui: Optional[Pingshui], modern_tone_fn) -> dict:
    """Tone facts about the last character of `line` (which must be non-empty)."""
    ch = line[-1]
    tone = modern_tone_fn(line)[-1]
    info = {
        "char": ch,
        "modern_tone": tone,
        "modern_class": class_of_tone(tone),
        "classical_tones": pingshui.tones(ch) if pingshui else "",
        "classical_class": pingshui.klass(ch) if pingshui else UNKNOWN,
    }
    return info


def _end_ok(info: dict, need: str) -> tuple[bool, bool, bool]:
    """(modern ok, classical ok, classical fell back to modern)."""
    modern_ok = info["modern_class"] == need
    classical = satisfies(info["classical_class"], need)
    if classical is None:
        return modern_ok, modern_ok, True
    return modern_ok, classical, False


def _even_opposition(c1: list[str], c2: list[str]) -> Optional[float]:
    """Share of even positions (2nd, 4th, …) where the classes are opposite.

    A BOTH class can stand opposite anything. A position with an UNKNOWN class
    counts as not opposite. None if there is no even position to compare.
    """
    idx = [i for i in range(1, min(len(c1), len(c2)), 2)]
    if not idx:
        return None
    ok = 0
    for i in idx:
        a, b = c1[i], c2[i]
        if UNKNOWN in (a, b):
            continue
        if BOTH in (a, b) or a != b:
            ok += 1
    return ok / len(idx)


def check_pair(
    first: str,
    second: str,
    judged: str,
    *,
    pingshui: Optional[Pingshui] = None,
    tagger: Tagger = jieba_tagger,
    coarse=jieba_coarse,
    modern_tone_fn=modern_tones,
    with_parallelism: bool = True,
) -> dict:
    """Check a couplet in which the line named by `judged` is the one under test.

    `judged` is "first" or "second". The other line is taken as given: it is
    never blamed for its own tone. Both lines may contain non-汉字; rule 1
    fails for a judged line that does.
    """
    if judged not in ("first", "second"):
        raise ValueError(f"judged must be 'first' or 'second', got {judged!r}")

    raw_judged = first if judged == "first" else second
    h1, h2 = hanzi_only(first), hanzi_only(second)
    hj = h1 if judged == "first" else h2
    other = h2 if judged == "first" else h1

    result: dict = {
        "judged": judged,
        "len_first": len(h1),
        "len_second": len(h2),
        "empty": not hj,
        "pure": is_pure_hanzi(raw_judged),
        "echo": bool(hj) and hj == other,
    }

    if not hj or not other:
        result.update(
            rule_length=False, rule_tone_modern=False, rule_tone_classical=False,
            tone_classical_fallback=False, rule_no_repeat=False, repeated="",
            n_same_position=0, all_rules_modern=False, all_rules_classical=False,
            end_char="", end_modern_tone=None, end_modern_class=UNKNOWN,
            end_classical_tones="", end_classical_class=UNKNOWN,
            even_opposition_modern=None, even_opposition_classical=None,
            pos_match=None, boundary_match=None,
        )
        return result

    # Rule 1: length.
    rule_length = result["pure"] and len(h1) == len(h2)

    # Rule 2: end tone of the judged line.
    info = _end_info(hj, pingshui, modern_tone_fn)
    modern_ok, classical_ok, fell_back = _end_ok(info, END_NEED[judged])

    # Rule 3: no repeated characters.
    repeated = repeated_characters(h1, h2)
    rule_no_repeat = not repeated

    not_echo = not result["echo"]
    result.update(
        rule_length=rule_length,
        rule_tone_modern=modern_ok,
        rule_tone_classical=classical_ok,
        tone_classical_fallback=fell_back,
        rule_no_repeat=rule_no_repeat,
        repeated="".join(repeated),
        n_same_position=same_position_count(h1, h2),
        all_rules_modern=rule_length and modern_ok and rule_no_repeat and not_echo,
        all_rules_classical=rule_length and classical_ok and rule_no_repeat and not_echo,
        end_char=info["char"],
        end_modern_tone=info["modern_tone"],
        end_modern_class=info["modern_class"],
        end_classical_tones=info["classical_tones"],
        end_classical_class=info["classical_class"],
    )

    # Scores that need the two lines to line up position by position.
    if len(h1) == len(h2):
        m1 = [class_of_tone(t) for t in modern_tone_fn(h1)]
        m2 = [class_of_tone(t) for t in modern_tone_fn(h2)]
        result["even_opposition_modern"] = _even_opposition(m1, m2)
        if pingshui is not None:
            # Fall back to the modern class where the table has no entry.
            k1 = [c if c != UNKNOWN else m for c, m in zip(pingshui.classes(h1), m1)]
            k2 = [c if c != UNKNOWN else m for c, m in zip(pingshui.classes(h2), m2)]
            result["even_opposition_classical"] = _even_opposition(k1, k2)
        else:
            result["even_opposition_classical"] = None
        if with_parallelism:
            result.update(parallelism(h1, h2, tagger=tagger, coarse=coarse))
        else:
            result.update(pos_match=None, boundary_match=None)
    else:
        result.update(
            even_opposition_modern=None, even_opposition_classical=None,
            pos_match=None, boundary_match=None,
        )
    return result


def check_generated(given: str, answer: str, direction: str, **kwargs) -> dict:
    """Score a generated line. `direction` is FORWARD or REVERSE."""
    if direction == FORWARD:
        return check_pair(given, answer, "second", **kwargs)
    if direction == REVERSE:
        return check_pair(answer, given, "first", **kwargs)
    raise ValueError(f"direction must be {FORWARD!r} or {REVERSE!r}, got {direction!r}")


def check_reference(first: str, second: str, **kwargs) -> dict[str, dict]:
    """Score an original couplet as if each line had been generated.

    Returns {"forward": ..., "reverse": ...}: forward judges the second line
    given the first, reverse judges the first given the second.
    """
    return {
        FORWARD: check_pair(first, second, "second", **kwargs),
        REVERSE: check_pair(first, second, "first", **kwargs),
    }


def reference_match(answer: str, reference: str) -> dict:
    """How close a generated line is to the original line (the memorization meter)."""
    a, r = hanzi_only(answer), hanzi_only(reference)
    if not a or not r:
        return {"exact": False, "same_position_share": 0.0, "char_jaccard": 0.0}
    same = sum(x == y for x, y in zip(a, r)) / max(len(a), len(r))
    sa, sr = set(a), set(r)
    return {
        "exact": a == r,
        "same_position_share": same,
        "char_jaccard": len(sa & sr) / len(sa | sr),
    }
