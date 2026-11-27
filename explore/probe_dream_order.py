"""Why does Dream fill its canvas almost left to right? Its sampler's ranking at step 0.

Usage:
    python explore/probe_dream_order.py

Post hoc (PROTOCOL.md entry 36). Probe L found that, read at temperature 1,
Dream is most certain about the end padding, yet its runs fixed the line first,
nearly left to right. Dream's entropy rule (generation_utils.sample_tokens)
computes the entropy *after* temperature scaling and top-p truncation, and
ranks positions with torch.topk. This recomputes that ranking at step 0 for
the phase 1 settings (temperature 0.5 and 0.2, top_p 0.95) on the fully masked
canvas, for a seeded sample of 300 canvases, and records:
  - which position the sampler would fix first (line or padding; leftmost?),
  - how many positions tie at the best score (entropy exactly 0 after truncation),
  - at padding positions, how the end mass splits between <|endoftext|> and <|im_end|>.
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dfl import models  # noqa: E402
from dfl.generate import chat_ids  # noqa: E402
from dfl.prompts import FORWARD, REVERSE, build_prompt  # noqa: E402
from dfl.text import hanzi_only  # noqa: E402

OUT = ROOT / "explore" / "out"
SLACK = 6


def top_p_filter(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    """Dream's top_p_logits, re-implemented: keep the smallest set with mass >= top_p."""
    sorted_logits, sorted_idx = torch.sort(logits, descending=True)
    cum = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
    remove = cum > top_p
    remove[..., 1:] = remove[..., :-1].clone()
    remove[..., 0] = False
    mask = torch.zeros_like(logits, dtype=torch.bool).scatter_(-1, sorted_idx, remove)
    return logits.masked_fill(mask, torch.finfo(logits.dtype).min)


@torch.no_grad()
def main() -> int:
    items = []
    for f in ["data/unseen/huohou.jsonl", "data/recall/recall.jsonl", "data/dev/dev.jsonl"]:
        for line in open(ROOT / f, encoding="utf-8"):
            items.append(json.loads(line))
    rng = random.Random(20261125)
    picks = rng.sample([(it, d) for it in items for d in (FORWARD, REVERSE)], 300)
    loaded = models.load("dream")
    model, tok, dev = loaded.model, loaded.tokenizer, loaded.device
    mask_id = model.config.mask_token_id
    eot = tok.convert_tokens_to_ids("<|endoftext|>")
    imend = tok.convert_tokens_to_ids("<|im_end|>")
    rows = []
    for it, direction in picks:
        given = it["first"] if direction == FORWARD else it["second"]
        n = len(hanzi_only(given))
        canvas = n + SLACK
        p_ids = chat_ids(loaded, build_prompt(given, direction, n))[0].tolist()
        x = torch.tensor([p_ids + [mask_id] * canvas], device=dev)
        P = len(p_ids)
        raw = model(x, "full", None).logits[0, P - 1:P - 1 + canvas]  # shifted, as Dream's sampler does; native dtype
        logits = raw.float()
        p1 = torch.softmax(logits, dim=-1)
        is_pad = (p1[:, eot] + p1[:, imend]) > 0.5
        rec = {"item_id": it["id"], "direction": direction, "canvas": canvas, "pad": is_pad.tolist(),
               "pad_split_eot": [round(float(p1[j, eot] / (p1[j, eot] + p1[j, imend])), 3) for j in range(canvas) if is_pad[j]]}
        for T in (0.5, 0.2):
            lg = top_p_filter(raw / T, 0.95)  # in the model's own dtype (bf16), as the sampler computes it
            probs = torch.softmax(lg, dim=-1)
            conf = torch.sum(probs * torch.log(probs + 1e-10), dim=-1)  # neg entropy, as in sample_tokens
            first = int(torch.topk(conf, 1).indices[0])
            best = float(conf.max())
            n_ties = int((conf >= best - 1e-6).sum())
            n_zero = int((conf > -1e-6).sum())
            rec[f"T{T}"] = {"first": first, "first_is_pad": bool(is_pad[first]), "first_is_leftmost": first == 0,
                            "n_ties_at_best": n_ties, "n_zero_entropy": n_zero,
                            "zero_in_line": int(((conf > -1e-6) & ~is_pad).sum()), "zero_in_pad": int(((conf > -1e-6) & is_pad).sum())}
        rows.append(rec)
    # shift check: on known couplets, the slot read from position i - 1 (Dream's convention) against position i
    recall = {}
    for line in open(ROOT / "data" / "recall" / "recall.jsonl", encoding="utf-8"):
        it = json.loads(line)
        recall[it["id"]] = it
    shift_lines = []
    for iid in ("hlm-001", "xyj-027"):
        it = recall[iid]
        n = len(hanzi_only(it["first"]))
        prefix = hanzi_only(it["second"])[:-1]
        p_ids = chat_ids(loaded, build_prompt(it["first"], FORWARD, n))[0].tolist()
        a_ids = tok(prefix, add_special_tokens=False).input_ids
        x = torch.tensor([p_ids + a_ids + [mask_id] + [eot] * (n + SLACK - len(a_ids) - 1)], device=dev)
        pos = len(p_ids) + len(a_ids)
        out = model(x, "full", None).logits[0].float()
        for name, j in (("read at i-1", pos - 1), ("read at i", pos)):
            pr = torch.softmax(out[j], dim=-1)
            top = torch.topk(pr, 4)
            shift_lines.append(f"{iid} {prefix}_ ({name}): " + " ".join(f"{tok.decode([int(k)])!r}:{float(v):.2f}" for v, k in zip(top.values, top.indices)))
    print("\n".join(shift_lines), flush=True)
    with open(OUT / "probe_dream_order.jsonl", "w", encoding="utf-8", newline="\n") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    lines = ["# Dream's sampler ranking at step 0\n",
             "Post hoc; see `PROTOCOL.md` entry 36 and `explore/probe_dream_order.py`. 300 seeded canvases, phase 1 prompt and canvas, "
             "Dream's own entropy score after temperature scaling and top-p 0.95 truncation.\n",
             "| Temperature | First pick in the line | First pick leftmost | Positions tied at the best score (mean) | Line positions with zero entropy (mean) | Padding positions with zero entropy (mean) |",
             "|---|---|---|---|---|---|"]
    for T in (0.5, 0.2):
        k = f"T{T}"
        lines.append(f"| {T} | {100 * sum(not r[k]['first_is_pad'] for r in rows) / len(rows):.0f}% | "
                     f"{100 * sum(r[k]['first_is_leftmost'] for r in rows) / len(rows):.0f}% | "
                     f"{sum(r[k]['n_ties_at_best'] for r in rows) / len(rows):.1f} | "
                     f"{sum(r[k]['zero_in_line'] for r in rows) / len(rows):.1f} | {sum(r[k]['zero_in_pad'] for r in rows) / len(rows):.1f} |")
    splits = [s for r in rows for s in r["pad_split_eot"]]
    lines.append("")
    lines.append(f"At padding positions (temperature 1), the end mass going to <|endoftext|> rather than <|im_end|>: mean {sum(splits) / max(1, len(splits)):.2f} "
                 f"over {len(splits)} positions; share of positions where neither takes 95% of it: "
                 f"{100 * sum(0.05 < s < 0.95 for s in splits) / max(1, len(splits)):.0f}%.\n")
    (OUT / "dream_order.md").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print("\n".join(lines), flush=True)
    models.unload(loaded)
    return 0


if __name__ == "__main__":
    sys.exit(main())
