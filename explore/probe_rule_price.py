"""What would it cost to enforce the end-tone rule in the decoder?

Usage:
    python explore/probe_rule_price.py qwen|llada

Post hoc (PROTOCOL.md entry 36), for prediction 3 in results/theory.md. Same
slot as probe S: the exact phase 1 prompt, the original line minus its last
character, and the slot that follows (for LLaDA with the end padding in place,
the state its sampler is in). Among tokens that are one unambiguous 汉字:
  - allowed = the class the rule asks for (平 for a second line, 仄 for a first);
  - share_top_allowed: the single most likely character is already allowed;
  - price: log P(best allowed character) - log P(best character), in nats
    (0 when the best is allowed; how much less likely the forced choice is).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "explore"))
sys.path.insert(0, str(ROOT / "src"))

from dfl import models  # noqa: E402
from dfl.prompts import FORWARD, REVERSE, build_prompt  # noqa: E402
from dfl.text import hanzi_only  # noqa: E402
from probe_tones import SLACK, load_items, single_hanzi_classes, slot_logits  # noqa: E402

OUT = ROOT / "explore" / "out"


@torch.no_grad()
def main() -> int:
    key = sys.argv[1]
    own = len(sys.argv) > 2 and sys.argv[2] == "own"  # prefix = the model's own first sample (bare, right length)
    items = load_items()
    first_sample = {}
    if own:
        with open(ROOT / "results" / "full_scored.jsonl", encoding="utf-8") as fh:
            for line in fh:
                r = json.loads(line)
                if r["model"] == key and r["condition"] == "bare" and r["sample"] == 0 and r.get("rule_length"):
                    first_sample[(r["item_id"], r["direction"])] = hanzi_only(r["answer"])
    loaded = models.load(key)
    ping_ids, ze_ids, _ = single_hanzi_classes(loaded.tokenizer)
    ping_ids, ze_ids = ping_ids.to(loaded.device), ze_ids.to(loaded.device)
    rows = []
    for it in items:
        for direction, given, written in ((FORWARD, it["first"], it["second"]), (REVERSE, it["second"], it["first"])):
            n = len(hanzi_only(given))
            line = first_sample.get((it["id"], direction)) if own else hanzi_only(written)
            if not line:
                continue
            lg = slot_logits(loaded, build_prompt(given, direction, n), line[:-1], n + SLACK)
            lp = torch.log_softmax(lg, dim=0)
            best_ping, best_ze = float(lp[ping_ids].max()), float(lp[ze_ids].max())
            allowed, other = (best_ping, best_ze) if direction == FORWARD else (best_ze, best_ping)
            best = max(allowed, other)
            rows.append({"item_id": it["id"], "set": it["set"], "direction": direction,
                         "top_allowed": allowed >= other, "price": allowed - best})
    lines = [f"## {key}, prefix = {'its own first sample (bare, right length)' if own else 'the original line'}\n","| Set | Direction | Best character already allowed | Mean price (nats) | Median price when not allowed |", "|---|---|---|---|---|"]
    for set_name in ("unseen", "recall", "dev"):
        for d in (FORWARD, REVERSE):
            sel = [r for r in rows if r["set"] == set_name and r["direction"] == d]
            if not sel:
                continue
            pr = np.array([r["price"] for r in sel])
            bad = pr[pr < 0]
            lines.append(f"| {set_name} | {d} | {100 * np.mean([r['top_allowed'] for r in sel]):.0f}% | {pr.mean():.2f} | "
                         f"{(np.median(bad) if len(bad) else 0):.2f} |")
    text = "\n".join(lines) + "\n"
    with open(OUT / "rule_price.md", "a", encoding="utf-8", newline="\n") as fh:
        fh.write(text + "\n")
    print(text, flush=True)
    models.unload(loaded)
    return 0


if __name__ == "__main__":
    sys.exit(main())
