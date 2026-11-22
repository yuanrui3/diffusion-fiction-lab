"""Score run files: extraction, the checker, match to the reference, relevance.

`score_record(rec, ...)` turns one run-file record (one item, direction and
condition, with its samples) into one scored row per sample.
`aggregate(rows, by=...)` turns rows into the table the reports use.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable, Optional, Sequence

from .checker import check_generated, reference_match
from .extract import extract_answer
from .prompts import wanted_line
from .tones import Pingshui

RULES = [
    "rule_length", "rule_tone_modern", "rule_tone_classical", "rule_no_repeat",
    "all_rules_modern", "all_rules_classical",
]
SCORES = ["pos_match", "boundary_match", "even_opposition_modern", "even_opposition_classical", "relevance"]


def default_tagger(device: Optional[str] = None):
    """The frozen tagger for the parallelism score: (tagger, tag-to-class map)."""
    import torch

    from .parallel import ltp_coarse, make_ltp_tagger
    from .registry import TAGGER, local_path

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return make_ltp_tagger(str(local_path(TAGGER)), device=device), ltp_coarse


def score_record(rec: dict, *, pingshui: Optional[Pingshui] = None, tagger=None, coarse=None) -> list[dict]:
    """One scored row per sample of a run-file item record."""
    rows = []
    want = wanted_line(rec["direction"])
    kwargs = {"pingshui": pingshui}
    if tagger is not None:
        kwargs["tagger"] = tagger
    if coarse is not None:
        kwargs["coarse"] = coarse
    for k, smp in enumerate(rec["samples"]):
        ext = extract_answer(smp["raw"], want)
        chk = check_generated(rec["given"], ext.answer, rec["direction"], **kwargs)
        ref = reference_match(ext.answer, rec["reference"])
        rows.append({
            "model": rec.get("label") or rec["model"],     # the arm: a model at one sampling setting
            "base_model": rec["model"],
            "item_id": rec["item_id"],
            "set": rec.get("set"),
            "direction": rec["direction"],
            "condition": rec["condition"],
            "sample": k,
            "given": rec["given"],
            "reference": rec["reference"],
            "raw": smp["raw"],
            "answer": ext.answer,
            "extract_notes": ext.notes,
            "junk_after_end": smp.get("junk_after_end", False),
            "unfilled": smp.get("unfilled", 0),
            **chk,
            "exact": ref["exact"],
            "ref_same_position": ref["same_position_share"],
            "ref_char_jaccard": ref["char_jaccard"],
            "relevance": None,
        })
    return rows


def add_relevance(rows: list[dict], summaries: dict[str, str], embedder) -> None:
    """Fill `relevance`: cosine similarity between the answer and the item's summary.

    `embedder` is anything with `encode(list_of_texts, normalize_embeddings=True)`
    returning an array (a sentence-transformers model). Rows whose item has no
    summary, or whose answer is empty, keep None.
    """
    todo = [r for r in rows if r["answer"] and summaries.get(r["item_id"])]
    if not todo:
        return
    answers = sorted({r["answer"] for r in todo})
    texts = sorted({summaries[r["item_id"]] for r in todo})
    a_vec = dict(zip(answers, embedder.encode(answers, normalize_embeddings=True, batch_size=64)))
    s_vec = dict(zip(texts, embedder.encode(texts, normalize_embeddings=True, batch_size=16)))
    for r in todo:
        r["relevance"] = float(a_vec[r["answer"]] @ s_vec[summaries[r["item_id"]]])


def _mean(values: Iterable) -> Optional[float]:
    vals = [float(v) for v in values if v is not None]
    return sum(vals) / len(vals) if vals else None


def aggregate(rows: Sequence[dict], by: Sequence[str]) -> list[dict]:
    """Per-group means. Rule columns are pass rates over all samples.

    Score columns are means over the samples where the score is defined;
    `pos_match_all` also counts an undefined parallelism score as 0, so a model
    cannot look more parallel by getting the length wrong.
    """
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        groups[tuple(r[k] for k in by)].append(r)

    table = []
    for key in sorted(groups, key=lambda k: tuple(str(x) for x in k)):
        g = groups[key]
        items = defaultdict(list)
        for r in g:
            items[r["item_id"]].append(r)
        row = dict(zip(by, key))
        row["n_items"] = len(items)
        row["n_samples"] = len(g)
        for rule in RULES:
            row[rule] = _mean(r[rule] for r in g)
        for score in SCORES:
            row[score] = _mean(r[score] for r in g)
        row["pos_match_all"] = _mean((r["pos_match"] or 0.0) for r in g)
        row["empty"] = _mean(r["empty"] for r in g)
        row["echo"] = _mean(r["echo"] for r in g)
        row["exact"] = _mean(r["exact"] for r in g)
        row["ref_same_position"] = _mean(r["ref_same_position"] for r in g)
        row["distinct_per_item"] = _mean(len({r["answer"] for r in rs}) for rs in items.values())
        row["any_all_rules_modern"] = _mean(any(r["all_rules_modern"] for r in rs) for rs in items.values())
        table.append(row)
    return table


def format_table(table: Sequence[dict], columns: Sequence[str]) -> str:
    """Plain-text table; rates as percentages, None as a dash."""
    def cell(row, col):
        v = row.get(col)
        if v is None:
            return "–"
        if isinstance(v, bool):
            return "yes" if v else "no"
        if isinstance(v, float):
            return f"{v:.2f}" if col in ("distinct_per_item", "relevance") else f"{100 * v:.0f}%"
        return str(v)

    cells = [[cell(r, c) for c in columns] for r in table]
    widths = [max(len(c), *(len(row[i]) for row in cells)) if cells else len(c) for i, c in enumerate(columns)]
    lines = ["  ".join(c.ljust(w) for c, w in zip(columns, widths))]
    lines += ["  ".join(v.ljust(w) for v, w in zip(row, widths)) for row in cells]
    return "\n".join(lines)
