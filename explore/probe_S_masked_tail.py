"""Probe S again for the diffusion models, with the rest of the canvas still masked.

Usage:
    python explore/probe_S_masked_tail.py dream|llada

Post hoc (PROTOCOL.md entry 36). Probe S placed end tokens right after the
final slot. That is the state LLaDA is in when it writes its last character (it
fixes the padding first), but not Dream's: Dream fills left to right and its
padding is usually still masked when the last character is chosen. Reading
Dream's slot in a state it rarely sees could make its distribution look flatter
than it is. Here the tail stays masked: canvas = prefix, slot, masks to the
phase 1 canvas size. Same items, prompts and outputs as probe S.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "explore"))
sys.path.insert(0, str(ROOT / "src"))

from dfl import models  # noqa: E402
from dfl.generate import chat_ids  # noqa: E402
from dfl.prompts import FORWARD, REVERSE, build_prompt  # noqa: E402
from dfl.text import hanzi_only  # noqa: E402
from probe_tones import SLACK, load_items, prompt_without_tone_rule, single_hanzi_classes  # noqa: E402

OUT = ROOT / "explore" / "out"


@torch.no_grad()
def slot_logits_masked_tail(loaded, prompt: str, prefix: str, total_canvas: int) -> torch.Tensor:
    tok, model, dev, kind = loaded.tokenizer, loaded.model, loaded.device, loaded.spec.kind
    mask_id = model.config.mask_token_id
    p_ids = chat_ids(loaded, prompt)[0].tolist()
    a_ids = tok(prefix, add_special_tokens=False).input_ids if prefix else []
    tail = [mask_id] * max(0, total_canvas - len(a_ids) - 1)
    x = torch.tensor([p_ids + a_ids + [mask_id] + tail], device=dev)
    pos = len(p_ids) + len(a_ids)
    if kind == "llada":
        return model(x).logits[0, pos].float()
    return model(x, "full", None).logits[0, pos - 1].float()


def main() -> int:
    key = sys.argv[1]
    items = load_items()
    loaded = models.load(key)
    ping_ids, ze_ids, char_of = single_hanzi_classes(loaded.tokenizer)
    ping_ids, ze_ids = ping_ids.to(loaded.device), ze_ids.to(loaded.device)
    ids_of = {}
    for t, c in char_of.items():
        ids_of.setdefault(c, []).append(t)
    rows = []
    t0 = time.time()
    for it in items:
        for direction, given, written in ((FORWARD, it["first"], it["second"]), (REVERSE, it["second"], it["first"])):
            n = len(hanzi_only(given))
            prefix = hanzi_only(written)[:-1]
            rec = {"item_id": it["id"], "set": it["set"], "direction": direction, "given": given, "written": written, "n_chars": n}
            for variant, prompt in (("rule", build_prompt(given, direction, n)), ("norule", prompt_without_tone_rule(given, direction, n))):
                pr = torch.softmax(slot_logits_masked_tail(loaded, prompt, prefix, n + SLACK), dim=0)
                m_ping, m_ze = float(pr[ping_ids].sum()), float(pr[ze_ids].sum())
                orig_ids = ids_of.get(hanzi_only(written)[-1], [])
                rec[variant] = {"p_ping": m_ping / max(1e-12, m_ping + m_ze), "hanzi_mass": m_ping + m_ze,
                                "p_original": float(pr[orig_ids].sum()) if orig_ids else None}
            rows.append(rec)
    with open(OUT / f"probe_S_{key}_maskedtail.jsonl", "w", encoding="utf-8", newline="\n") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    import numpy as np
    for variant in ("rule", "norule"):
        f = np.mean([r[variant]["p_ping"] for r in rows if r["direction"] == FORWARD])
        b = np.mean([r[variant]["p_ping"] for r in rows if r["direction"] == REVERSE])
        po = np.mean([r[variant]["p_original"] for r in rows if r[variant]["p_original"] is not None])
        hm = np.mean([r[variant]["hanzi_mass"] for r in rows])
        print(f"{key} masked tail, {variant}: P(平) second lines {f:.3f}, first lines {b:.3f}, P(original) {po:.3f}, mass on single 汉字 {hm:.2f}", flush=True)
    for d in (FORWARD, REVERSE):
        diff = np.mean([r["rule"]["p_ping"] - r["norule"]["p_ping"] for r in rows if r["direction"] == d])
        print(f"{key} masked tail, effect of stating the rule, {d}: {100 * diff:+.1f} points", flush=True)
    print(f"done in {time.time() - t0:.0f} s", flush=True)
    models.unload(loaded)
    return 0


if __name__ == "__main__":
    sys.exit(main())
