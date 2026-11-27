"""Two follow-up checks on the tone probes.

Usage:
    python explore/probe_checks.py qwen|llada|dream

Post hoc (PROTOCOL.md entry 36).

K2, a fairer tone-knowledge question. In probe K, Qwen put about 1% of its
    first-token mass on the digits 1–4: it wants to answer in words. Here the
    reply is started for the model with "第" and the next slot is read over both
    the digits and the Chinese numerals 一二三四, summed per tone.

Sanity. Top candidates for the final slot on a few well-known couplets, to check
    that the canvas construction of probe S (in particular Dream's one-position
    shift) behaves as expected.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "explore"))
sys.path.insert(0, str(ROOT / "src"))

from dfl import models  # noqa: E402
from dfl.generate import chat_ids  # noqa: E402
from dfl.prompts import FORWARD, REVERSE, build_prompt  # noqa: E402
from dfl.text import hanzi_only  # noqa: E402
from probe_tones import slot_logits  # noqa: E402

OUT = ROOT / "explore" / "out"


@torch.no_grad()
def reply_slot(loaded, prompt: str, reply_prefix: str, canvas: int = 4) -> torch.Tensor:
    """Logits for the slot right after `reply_prefix` at the start of the reply."""
    tok, model, dev, kind = loaded.tokenizer, loaded.model, loaded.device, loaded.spec.kind
    p_ids = chat_ids(loaded, prompt)[0].tolist()
    r_ids = tok(reply_prefix, add_special_tokens=False).input_ids
    if kind == "ar":
        return model(input_ids=torch.tensor([p_ids + r_ids], device=dev)).logits[0, -1].float()
    mask_id = model.config.mask_token_id
    x = torch.tensor([p_ids + r_ids + [mask_id] * canvas], device=dev)
    pos = len(p_ids) + len(r_ids)
    if kind == "llada":
        return model(x).logits[0, pos].float()
    return model(x, "full", None).logits[0, pos - 1].float()


def main() -> int:
    key = sys.argv[1]
    loaded = models.load(key)
    tok = loaded.tokenizer

    # ---------------------------------------------------------------- K2
    def ids_for(s):
        ids = tok(s, add_special_tokens=False).input_ids
        return ids[0] if len(ids) == 1 else None
    # every single-token way to start the tone's name: digit, numeral, numeral + 声
    per_tone = {t: sorted({i for i in (ids_for(d), ids_for(c), ids_for(c + "声"), ids_for(d + "声")) if i is not None})
                for t, (d, c) in {1: ("1", "一"), 2: ("2", "二"), 3: ("3", "三"), 4: ("4", "四")}.items()}
    print(f"K2 candidate tokens: { {t: [tok.decode([i]) for i in ids] for t, ids in per_tone.items()} }", flush=True)
    rows = [json.loads(l) for l in open(OUT / f"probe_K_{key}.jsonl", encoding="utf-8")]
    out = []
    for r in rows:
        lg = reply_slot(loaded, f"汉字“{r['char']}”在普通话里读第几声？", "第")
        pr = torch.softmax(lg, dim=0)
        mass = [float(pr[ids].sum()) for t, ids in sorted(per_tone.items())]
        tot = sum(mass)
        out.append({"char": r["char"], "tone": r["tone"], "p": [m / tot for m in mass], "numeral_mass": tot})
    with open(OUT / f"probe_K2_{key}.jsonl", "w", encoding="utf-8", newline="\n") as fh:
        for r in out:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    acc = sum(max(range(4), key=lambda i: r["p"][i]) + 1 == r["tone"] for r in out) / len(out)
    pz = sum((r["p"][0] + r["p"][1] > 0.5) == (r["tone"] <= 2) for r in out) / len(out)
    nm = sum(r["numeral_mass"] for r in out) / len(out)
    print(f"K2 {key}: tone {acc:.2f}, 平/仄 {pz:.2f}, mass on numerals {nm:.2f}", flush=True)

    # ---------------------------------------------------------------- sanity
    items = {}
    for line in open(ROOT / "data" / "recall" / "recall.jsonl", encoding="utf-8"):
        it = json.loads(line)
        items[it["id"]] = it
    for iid in ("hlm-001", "sgyy-001", "shz-010", "xyj-027"):
        it = items[iid]
        for direction, given, written in ((FORWARD, it["first"], it["second"]), (REVERSE, it["second"], it["first"])):
            n = len(hanzi_only(given))
            lg = slot_logits(loaded, build_prompt(given, direction, n), hanzi_only(written)[:-1], n + 6)
            pr = torch.softmax(lg, dim=0)
            top = torch.topk(pr, 5)
            cands = " ".join(f"{tok.decode([int(i)])!r}:{float(p):.2f}" for p, i in zip(top.values, top.indices))
            print(f"  {iid} {direction}: {hanzi_only(written)[:-1]}_ (original {hanzi_only(written)[-1]}) -> {cands}", flush=True)
    models.unload(loaded)
    return 0


if __name__ == "__main__":
    sys.exit(main())
