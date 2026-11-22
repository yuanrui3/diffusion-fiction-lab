"""Word-class parallelism (对仗) between two lines, position by position.

Each line is segmented and tagged. Every character takes the coarse word class
of the word it belongs to. The score is the share of character positions at
which the two lines carry the same coarse class. Word-boundary agreement is
reported beside it, because a couplet's rhythm (2-2-3, 3-2-2 and so on) is part
of its parallelism and a tagger can get classes right while cutting differently.

The tagger is pluggable: any callable that turns a line into (word, tag) pairs,
together with a function mapping its tags to the coarse classes below.
"""

from __future__ import annotations

from typing import Callable, Optional, Sequence

Tagged = Sequence[tuple[str, str]]
Tagger = Callable[[str], Tagged]

# Coarse classes: N noun-like, R pronoun, V verb, A adjective, D adverb,
# M numeral, Q measure word, F direction/position word, X function word, O other.
_JIEBA_EXACT = {
    "f": "F",
    "m": "M", "mq": "M", "mg": "M",
    "q": "Q", "qv": "Q", "qt": "Q",
    "d": "D", "dg": "D", "df": "D",
    "t": "N", "tg": "N", "s": "N",
    "z": "A", "b": "A", "zg": "A",
    "i": "O", "l": "O", "j": "N",
    "eng": "O", "x": "O",
}
_JIEBA_PREFIX = {
    "n": "N",
    "r": "R",
    "v": "V",
    "a": "A",
    "p": "X", "c": "X", "u": "X", "y": "X", "e": "X", "o": "X", "h": "X", "k": "X",
}


def jieba_coarse(tag: str) -> str:
    """Map a jieba.posseg tag to a coarse class."""
    if tag in _JIEBA_EXACT:
        return _JIEBA_EXACT[tag]
    return _JIEBA_PREFIX.get(tag[:1], "O")


_jieba_ready = False


def jieba_tagger(line: str) -> list[tuple[str, str]]:
    """Segment and tag one line with jieba.posseg (HMM on, default dictionary)."""
    global _jieba_ready
    import jieba
    import jieba.posseg as pseg

    if not _jieba_ready:
        import logging

        jieba.setLogLevel(logging.WARNING)
        _jieba_ready = True
    return [(p.word, p.flag) for p in pseg.lcut(line)]


def char_classes(tagged: Tagged, coarse: Callable[[str], str] = jieba_coarse) -> list[str]:
    """Coarse class of each character, in order."""
    out: list[str] = []
    for word, tag in tagged:
        out.extend([coarse(tag)] * len(word))
    return out


def boundaries(tagged: Tagged) -> set[int]:
    """Positions after which a word ends, not counting the end of the line."""
    cuts: set[int] = set()
    pos = 0
    for word, _ in tagged:
        pos += len(word)
        cuts.add(pos)
    cuts.discard(pos)
    return cuts


def parallelism(
    first: str,
    second: str,
    *,
    tagger: Tagger = jieba_tagger,
    coarse: Callable[[str], str] = jieba_coarse,
) -> dict[str, Optional[float]]:
    """Position-by-position word-class agreement between two lines.

    Returns `pos_match` and `boundary_match`, both in [0, 1], or None for both
    when the lines differ in length or are empty (position-by-position
    comparison has no meaning then).
    """
    if not first or len(first) != len(second):
        return {"pos_match": None, "boundary_match": None}

    t1, t2 = tagger(first), tagger(second)
    c1, c2 = char_classes(t1, coarse), char_classes(t2, coarse)
    if len(c1) != len(first) or len(c2) != len(second):
        # The tagger dropped or added characters; do not guess.
        return {"pos_match": None, "boundary_match": None}

    pos_match = sum(a == b for a, b in zip(c1, c2)) / len(first)

    b1, b2 = boundaries(t1), boundaries(t2)
    union = b1 | b2
    boundary_match = 1.0 if not union else len(b1 & b2) / len(union)

    return {"pos_match": pos_match, "boundary_match": boundary_match}


# ----------------------------------------------------------------------------
# Other taggers, tried against jieba in scripts/compare_taggers.py
# ----------------------------------------------------------------------------

def jieba_char_tagger(line: str) -> list[tuple[str, str]]:
    """One character per unit, tagged with the character's own entry in
    jieba's dictionary (no segmentation at all). Unknown characters get "x"."""
    import jieba.posseg as pseg

    if not pseg.dt.word_tag_tab:
        pseg.dt.initialize()
    table = pseg.dt.word_tag_tab
    return [(ch, table.get(ch, "x")) for ch in line]


# Universal POS tags (Stanza) to coarse classes.
_UPOS = {
    "NOUN": "N", "PROPN": "N", "PRON": "R", "VERB": "V", "AUX": "V", "ADJ": "A",
    "ADV": "D", "NUM": "M", "ADP": "X", "CCONJ": "X", "SCONJ": "X", "PART": "X",
    "DET": "R", "INTJ": "X", "PUNCT": "O", "SYM": "O", "X": "O",
}


def upos_coarse(tag: str) -> str:
    return _UPOS.get(tag, "O")


# LTP's 863 tag set to coarse classes.
_LTP = {
    "n": "N", "nh": "N", "ns": "N", "ni": "N", "nl": "N", "nt": "N", "nz": "N", "j": "N",
    "nd": "F", "r": "R", "v": "V", "a": "A", "b": "A", "z": "A", "d": "D", "m": "M", "q": "Q",
    "p": "X", "c": "X", "u": "X", "e": "X", "o": "X", "h": "X", "k": "X",
}


def ltp_coarse(tag: str) -> str:
    return _LTP.get(tag, "O")


def make_ltp_tagger(model_path: str, device: str = "cpu") -> Tagger:
    """Tagger backed by an LTP model folder (segmentation + 863 POS tags)."""
    from functools import lru_cache

    from ltp import LTP

    model = LTP(model_path)
    if device != "cpu":
        model.to(device)

    @lru_cache(maxsize=200_000)
    def tag(line: str) -> tuple[tuple[str, str], ...]:
        out = model.pipeline([line], tasks=["cws", "pos"])
        return tuple(zip(out.cws[0], out.pos[0]))

    return lambda line: list(tag(line))


def make_stanza_tagger(lang: str, model_dir: str, use_gpu: bool = False) -> Tagger:
    """Tagger backed by a Stanza pipeline ("zh-hans" modern, "lzh" classical)."""
    from functools import lru_cache

    import stanza

    nlp = stanza.Pipeline(
        lang, dir=model_dir, processors="tokenize,pos", tokenize_no_ssplit=True,
        use_gpu=use_gpu, download_method=None, verbose=False,
    )

    @lru_cache(maxsize=200_000)
    def tag(line: str) -> tuple[tuple[str, str], ...]:
        doc = nlp(line)
        return tuple((w.text, w.upos) for s in doc.sentences for w in s.words)

    return lambda line: list(tag(line))
