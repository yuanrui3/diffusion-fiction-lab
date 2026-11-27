"""What the diffusion models believe about the line before the first step.

Usage:
    python explore/probe_length.py llada
    python explore/probe_length.py dream

Post hoc and exploratory (PROTOCOL.md entry 36). One forward pass per item on
the fully masked phase 1 canvas (required characters + 6 tokens), with the
exact phase 1 prompt (v2, bare), both directions, all 723 items. For every
canvas position this records:
  - p_end: the probability of an end token (end-of-text or end-of-turn),
  - entropy: the entropy of the position's distribution,
  - conf: the largest probability (what LLaDA's low-confidence remasking ranks by).
From these: the length the model expects (the first position whose p_end
exceeds 0.5), which position would be fixed first under each sampler's rule,
and how sharply the model separates line from padding.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dfl import models  # noqa: E402
from dfl.generate import chat_ids, end_token_ids  # noqa: E402
from dfl.prompts import FORWARD, REVERSE, build_prompt  # noqa: E402
from dfl.text import hanzi_only  # noqa: E402

OUT = ROOT / "explore" / "out"
SLACK = 6


@torch.no_grad()
def main() -> int:
    key = sys.argv[1]
    items = []
    for f in ["data/unseen/huohou.jsonl", "data/recall/recall.jsonl", "data/dev/dev.jsonl"]:
        for line in open(ROOT / f, encoding="utf-8"):
            items.append(json.loads(line))
    loaded = models.load(key)
    model, tok, dev, kind = loaded.model, loaded.tokenizer, loaded.device, loaded.spec.kind
    mask_id = model.config.mask_token_id
    ends = sorted(end_token_ids(loaded))
    rows = []
    t0 = time.time()
    for it in items:
        for direction, given, written in ((FORWARD, it["first"], it["second"]), (REVERSE, it["second"], it["first"])):
            n = len(hanzi_only(given))
            canvas = n + SLACK
            p_ids = chat_ids(loaded, build_prompt(given, direction, n))[0].tolist()
            x = torch.tensor([p_ids + [mask_id] * canvas], device=dev)
            P = len(p_ids)
            if kind == "llada":
                logits = model(x).logits[0, P:P + canvas].float()
            else:
                logits = model(x, "full", None).logits[0, P - 1:P - 1 + canvas].float()  # Dream: shifted by one
            pr = torch.softmax(logits, dim=-1)
            p_end = pr[:, ends].sum(-1)
            ent = -(pr * torch.log(pr.clamp_min(1e-30))).sum(-1)
            conf = pr.max(-1).values
            # the reference line's length in this model's tokens, for comparison
            ref_tokens = len(tok(hanzi_only(written), add_special_tokens=False).input_ids)
            rows.append({"item_id": it["id"], "set": it["set"], "direction": direction, "n_chars": n,
                         "canvas": canvas, "ref_tokens": ref_tokens,
                         "p_end": [round(float(v), 4) for v in p_end],
                         "entropy": [round(float(v), 3) for v in ent],
                         "conf": [round(float(v), 4) for v in conf]})
    with open(OUT / f"probe_L_{key}.jsonl", "w", encoding="utf-8", newline="\n") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    print(f"probe L: {len(rows)} canvases in {time.time() - t0:.0f} s", flush=True)
    models.unload(loaded)
    return 0


if __name__ == "__main__":
    sys.exit(main())
