"""Two probes of tone, read straight from the models' output distributions (no sampling).

Usage:
    python explore/probe_tones.py qwen      # one model per process, as for generation
    python explore/probe_tones.py llada
    python explore/probe_tones.py dream

Post hoc and exploratory (PROTOCOL.md entry 36). Nothing frozen is changed.

Probe K, knowledge: does the model know a character's Mandarin tone?
    "汉字“X”在普通话里读第几声？只回答一个数字。" The reply's first position is
    read as a distribution over the tokens 1, 2, 3, 4. Characters: the final
    characters of every original line (all three sets) whose modern reading is
    unambiguous.

Probe S, the final slot: when everything but the last character is in place,
    does the model's choice for that slot depend on the rule?
    The exact phase 1 prompt (v2, bare), followed by the original line minus its
    last character; the next slot's distribution is restricted to tokens that are
    one unambiguous 汉字, and the mass on 平 is recorded. Run twice: with the
    prompt as used (rule 三 states the end tone) and with rule 三 removed. Both
    directions, every item of the unseen, recall and dev sets.
    - Qwen: the next-token distribution after the prefix.
    - LLaDA: canvas = prefix, one mask, end-of-turn, end-of-text padding to the
      phase 1 canvas size; the mask's distribution.
    - Dream: the same canvas with its own end token; Dream's logits are shifted
      by one position (its sampler reads position i from the output at i - 1).
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
from dfl.generate import chat_ids  # noqa: E402
from dfl.prompts import FORWARD, REVERSE, _NUMERALS, _ROLE, build_prompt, chinese_numeral  # noqa: E402
from dfl.text import hanzi_only, is_hanzi  # noqa: E402
from dfl.tones import PING, ZE, modern_classes, modern_is_ambiguous, modern_tones  # noqa: E402

OUT = ROOT / "explore" / "out"
SLACK = 6  # the phase 1 canvas slack (config/phase1.json)


def prompt_without_tone_rule(given: str, direction: str, n_chars: int) -> str:
    """The v2 bare prompt with rule 三 (the end tone) removed and the rest renumbered."""
    r = _ROLE[direction]
    n = chinese_numeral(n_chars)
    intro = f"下面是一部章回体小说某一回回目的{r['given']}。{r['verb']}"
    material = f"{r['given']}：{given}"
    rules = [
        f"{r['want']}必须正好{n}个字，与{r['given']}字数相同",
        f"与{r['given']}对仗，词性、节奏逐字相对",
        f"不要重复{r['given']}用过的字",
    ]
    body = "\n".join(f"{_NUMERALS[i]}、{rule}{'；' if i < len(rules) else '。'}" for i, rule in enumerate(rules, 1))
    tail = f"只写出{r['want']}本身，不加标点，不作解释。"
    return f"{intro}\n\n要求：\n{body}\n\n{tail}\n\n{material}\n{r['want']}："


def single_hanzi_classes(tok) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """Ids of tokens that decode to exactly one unambiguous 汉字, split by modern class."""
    ping, ze, char_of = [], [], {}
    for tid in range(len(tok)):
        try:
            s = tok.decode([tid])
        except Exception:
            continue
        if len(s) != 1 or not is_hanzi(s) or modern_is_ambiguous(s):
            continue
        c = modern_classes(s)[0]
        if c == PING:
            ping.append(tid)
        elif c == ZE:
            ze.append(tid)
        else:
            continue
        char_of[tid] = s
    return torch.tensor(ping), torch.tensor(ze), char_of


def load_items():
    items = []
    for f in ["data/unseen/huohou.jsonl", "data/recall/recall.jsonl", "data/dev/dev.jsonl"]:
        for line in open(ROOT / f, encoding="utf-8"):
            items.append(json.loads(line))
    return items


@torch.no_grad()
def slot_logits(loaded, prompt: str, prefix: str, total_canvas: int) -> torch.Tensor:
    """Logits (vocab,) for the slot right after `prefix` in the reply."""
    tok, model, dev, kind = loaded.tokenizer, loaded.model, loaded.device, loaded.spec.kind
    p_ids = chat_ids(loaded, prompt)[0].tolist()
    a_ids = tok(prefix, add_special_tokens=False).input_ids if prefix else []
    if kind == "ar":
        x = torch.tensor([p_ids + a_ids], device=dev)
        return model(input_ids=x).logits[0, -1].float()
    mask_id = model.config.mask_token_id
    if kind == "llada":
        eot = tok.convert_tokens_to_ids("<|eot_id|>")
        eos = tok.convert_tokens_to_ids("<|endoftext|>")
        tail = [eot] + [eos] * max(0, total_canvas - len(a_ids) - 2)
        x = torch.tensor([p_ids + a_ids + [mask_id] + tail], device=dev)
        pos = len(p_ids) + len(a_ids)
        return model(x).logits[0, pos].float()
    if kind == "dream":
        eos = tok.convert_tokens_to_ids("<|endoftext|>")
        tail = [eos] * max(0, total_canvas - len(a_ids) - 1)
        x = torch.tensor([p_ids + a_ids + [mask_id] + tail], device=dev)
        pos = len(p_ids) + len(a_ids)
        logits = model(x, "full", None).logits
        return logits[0, pos - 1].float()  # Dream reads position i from the output at i - 1
    raise ValueError(kind)


@torch.no_grad()
def answer_logits(loaded, prompt: str, canvas: int = 4) -> torch.Tensor:
    """Logits for the first position of the reply."""
    model, dev, kind = loaded.model, loaded.device, loaded.spec.kind
    p_ids = chat_ids(loaded, prompt)[0].tolist()
    if kind == "ar":
        return model(input_ids=torch.tensor([p_ids], device=dev)).logits[0, -1].float()
    mask_id = model.config.mask_token_id
    x = torch.tensor([p_ids + [mask_id] * canvas], device=dev)
    if kind == "llada":
        return model(x).logits[0, len(p_ids)].float()
    return model(x, "full", None).logits[0, len(p_ids) - 1].float()


def main() -> int:
    key = sys.argv[1]
    OUT.mkdir(parents=True, exist_ok=True)
    items = load_items()

    # sanity: the "with rule" prompt is exactly the phase 1 prompt
    it = items[0]
    assert build_prompt(it["first"], FORWARD, len(hanzi_only(it["first"]))) == build_prompt(it["first"], FORWARD, len(hanzi_only(it["first"])), version="v2")

    t0 = time.time()
    loaded = models.load(key)
    tok = loaded.tokenizer
    print(f"loaded {key} in {time.time() - t0:.0f} s", flush=True)
    ping_ids, ze_ids, char_of = single_hanzi_classes(tok)
    print(f"single-hanzi tokens: {len(ping_ids)} 平, {len(ze_ids)} 仄", flush=True)
    ping_ids, ze_ids = ping_ids.to(loaded.device), ze_ids.to(loaded.device)
    both = torch.cat([ping_ids, ze_ids])
    ids_of = {}
    for t, c in char_of.items():
        ids_of.setdefault(c, []).append(t)

    # ---------------------------------------------------------------- probe K
    chars = {}
    for it in items:
        for line in (it["first"], it["second"]):
            h = hanzi_only(line)
            if h and not modern_is_ambiguous(h[-1]):
                t = modern_tones(h[-1])[0]
                if t in (1, 2, 3, 4):
                    chars[h[-1]] = t
    digit_ids = [tok.convert_tokens_to_ids(d) for d in "1234"]
    assert all(isinstance(d, int) and tok.decode([d]) == s for d, s in zip(digit_ids, "1234")), digit_ids
    k_rows = []
    for ch, tone in sorted(chars.items()):
        lg = answer_logits(loaded, f"汉字“{ch}”在普通话里读第几声？只回答一个数字。")
        p = torch.softmax(lg[digit_ids], dim=0).tolist()
        mass = float(torch.softmax(lg, dim=0)[digit_ids].sum())
        k_rows.append({"char": ch, "tone": tone, "p": p, "digit_mass": mass})
    with open(OUT / f"probe_K_{key}.jsonl", "w", encoding="utf-8", newline="\n") as fh:
        for r in k_rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    acc = sum(max(range(4), key=lambda i: r["p"][i]) + 1 == r["tone"] for r in k_rows) / len(k_rows)
    pz = sum((r["p"][0] + r["p"][1] > 0.5) == (r["tone"] <= 2) for r in k_rows) / len(k_rows)
    print(f"probe K: {len(k_rows)} characters, tone accuracy {acc:.2f}, 平/仄 accuracy {pz:.2f}", flush=True)

    # ---------------------------------------------------------------- probe S
    s_rows = []
    t1 = time.time()
    for n_done, it in enumerate(items):
        for direction, given, written in ((FORWARD, it["first"], it["second"]), (REVERSE, it["second"], it["first"])):
            n_chars = len(hanzi_only(given))
            prefix = hanzi_only(written)[:-1]
            canvas = n_chars + SLACK
            rec = {"item_id": it["id"], "set": it["set"], "direction": direction, "given": given,
                   "written": written, "n_chars": n_chars}
            for variant, prompt in (("rule", build_prompt(given, direction, n_chars)),
                                    ("norule", prompt_without_tone_rule(given, direction, n_chars))):
                lg = slot_logits(loaded, prompt, prefix, canvas)
                pr = torch.softmax(lg, dim=0)
                m_ping, m_ze = float(pr[ping_ids].sum()), float(pr[ze_ids].sum())
                sub = torch.softmax(lg[both], dim=0)
                ent = float(-(sub * torch.log(sub.clamp_min(1e-30))).sum())
                top = int(both[int(sub.argmax())])
                orig = hanzi_only(written)[-1]
                orig_ids = ids_of.get(orig, [])
                p_orig = float(pr[orig_ids].sum()) if orig_ids else None
                rec[variant] = {"p_ping": m_ping / max(1e-12, m_ping + m_ze), "hanzi_mass": m_ping + m_ze,
                                "entropy": ent, "top": char_of[top], "p_original": p_orig}
            s_rows.append(rec)
        if n_done % 100 == 0:
            print(f"  probe S {n_done}/{len(items)} items, {time.time() - t1:.0f} s", flush=True)
    with open(OUT / f"probe_S_{key}.jsonl", "w", encoding="utf-8", newline="\n") as fh:
        for r in s_rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"probe S: {len(s_rows)} slots in {time.time() - t1:.0f} s", flush=True)
    models.unload(loaded)
    return 0


if __name__ == "__main__":
    sys.exit(main())
