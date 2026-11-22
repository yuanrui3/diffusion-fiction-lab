"""Generation behind one interface for the three kinds of model.

    out = generate(loaded, user_message, n_samples=8, n_chars=7, seed=..., settings=GenSettings())

- "ar"    Qwen: ordinary left-to-right sampling with `model.generate`, at the
          model's own default sampling settings unless overridden.
- "llada" LLaDA: a port of the authors' reference sampler (`generate.py` in
          ML-GSAI/LLaDA, MIT licence), with the same Gumbel-max sampling in
          float64 and low-confidence remasking. Differences from the reference:
          a whole batch of samples shares one prompt; the output head is run on
          the canvas positions only (same numbers, less memory); the order in
          which positions are committed is recorded; the unmasking order can be
          switched (for phase 3); and tokens can be fixed in the canvas
          beforehand (anchors, for phase 2).
- "dream" Dream: the authors' own `diffusion_generate`, called as their README
          shows. The canvas history it returns gives the commit order. Anchors
          are written through the hook the authors provide for that purpose.

A diffusion reply is the canvas up to its first end token; whatever follows is
dropped and flagged (`junk_after_end`). The autoregressive reply ends at its
first end token by construction.
"""

from __future__ import annotations

import hashlib
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from typing import Optional

import torch
import torch.nn.functional as F

from .models import Loaded

ORDERS = ("confidence", "random", "l2r", "r2l")


@dataclass
class GenSettings:
    # Diffusion canvas: required characters + slack, in tokens.
    canvas_slack: int = 4
    # Diffusion steps; None means one step per canvas token.
    steps: Optional[int] = None
    # Sampling temperature; None means the model's documented default
    # (Qwen: its generation_config; LLaDA: 0; Dream: 0.2).
    temperature: Optional[float] = None
    # Nucleus cut-off; None means the documented default (Dream: 0.95; LLaDA has none).
    top_p: Optional[float] = None
    # LLaDA unmasking order: "confidence" is the authors' low-confidence remasking.
    order: str = "confidence"
    # Dream ordering algorithm and its temperature, as in the README.
    dream_alg: str = "entropy"
    dream_alg_temp: float = 0.0
    # Autoregressive budget; generous, so a compliant reply is never cut.
    ar_max_new_tokens: int = 64
    # How many samples go through the model together. Smaller batches need less
    # GPU memory; the samples are the same kind either way. Each batch after the
    # first gets its own seed, derived from the item seed.
    samples_per_batch: int = 4


@dataclass
class Sample:
    raw: str                                  # decoded reply, cut at the first end token
    token_ids: list[int]                      # generated ids (the whole canvas for diffusion)
    commit_step: Optional[list[int]] = None   # diffusion: step at which each canvas position was filled (-1: fixed beforehand)
    junk_after_end: bool = False              # diffusion: real tokens after the first end token
    unfilled: int = 0                         # diffusion: mask tokens left in the canvas


@dataclass
class GenOutput:
    samples: list[Sample]
    settings: dict
    seconds: float
    n_forward: int
    prompt_tokens: int
    peak_gpu_gib: Optional[float] = None        # peak memory held by tensors
    peak_reserved_gib: Optional[float] = None   # peak memory PyTorch took from the driver


def item_seed(base_seed: int, *parts: str) -> int:
    """A stable 31-bit seed from a base seed and identifying strings."""
    digest = hashlib.sha256("|".join([str(base_seed), *parts]).encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big") & 0x7FFFFFFF


def chat_ids(loaded: Loaded, user_message: str) -> torch.Tensor:
    """Token ids of a one-message chat, ready for the reply. Shape (1, P), on the CPU."""
    tok = loaded.tokenizer
    text = tok.apply_chat_template(
        [{"role": "user", "content": user_message}], tokenize=False, add_generation_prompt=True
    )
    return tok(text, add_special_tokens=False, return_tensors="pt").input_ids


_END_NAMES = {
    "llada": ["<|endoftext|>", "<|eot_id|>"],
    "dream": ["<|endoftext|>", "<|im_end|>"],
    "ar": ["<|endoftext|>", "<|im_end|>"],
}


def end_token_ids(loaded: Loaded) -> set[int]:
    """Ids that end a reply: the model's end-of-turn and end-of-text tokens."""
    tok = loaded.tokenizer
    ids: set[int] = set()
    for name in _END_NAMES[loaded.spec.kind]:
        tid = tok.convert_tokens_to_ids(name)
        # Keep it only if the tokenizer really has this token (not an unknown-token fallback).
        if tid is not None and tok.convert_ids_to_tokens(tid) == name:
            ids.add(tid)
    if tok.eos_token_id is not None:
        ids.add(tok.eos_token_id)
    return ids


def _decode(loaded: Loaded, ids: list[int], ends: set[int], mask_id: Optional[int]) -> Sample:
    cut = next((i for i, t in enumerate(ids) if t in ends), len(ids))
    body = [t for t in ids[:cut] if t != mask_id]
    junk = any(t not in ends and t != mask_id for t in ids[cut:])
    unfilled = sum(t == mask_id for t in ids) if mask_id is not None else 0
    text = loaded.tokenizer.decode(body, skip_special_tokens=True)
    return Sample(raw=text, token_ids=list(ids), junk_after_end=junk, unfilled=unfilled)


# ----------------------------------------------------------------------------
# Autoregressive
# ----------------------------------------------------------------------------

@torch.no_grad()
def _ar_sample(loaded: Loaded, prompt_ids: torch.Tensor, n: int, seed: int, s: GenSettings):
    model, dev = loaded.model, loaded.device
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    ids = prompt_ids.to(dev)
    overrides = {}
    if s.temperature is not None:
        overrides.update(do_sample=s.temperature > 0, temperature=s.temperature if s.temperature > 0 else None)
    if s.top_p is not None:
        overrides["top_p"] = s.top_p
    out = model.generate(
        input_ids=ids,
        attention_mask=torch.ones_like(ids),
        max_new_tokens=s.ar_max_new_tokens,
        num_return_sequences=n,
        **overrides,
    )
    new = out[:, ids.shape[1]:]
    gc = model.generation_config
    effective = {
        "do_sample": overrides.get("do_sample", gc.do_sample),
        "temperature": overrides.get("temperature", gc.temperature),
        "top_p": overrides.get("top_p", gc.top_p),
        "top_k": gc.top_k,
        "repetition_penalty": gc.repetition_penalty,
        "max_new_tokens": s.ar_max_new_tokens,
    }
    ends = end_token_ids(loaded)
    samples = [_decode(loaded, row.tolist(), ends, None) for row in new]
    return samples, effective, new.shape[1]


# ----------------------------------------------------------------------------
# LLaDA
# ----------------------------------------------------------------------------

@contextmanager
def _head_on_canvas_only(model, start: int):
    """Run LLaDA's output head on positions >= start only."""
    head = model.model.transformer.ff_out
    handle = head.register_forward_pre_hook(lambda module, args: (args[0][:, start:, :],))
    try:
        yield
    finally:
        handle.remove()


@torch.no_grad()
def _llada_sample(
    loaded: Loaded, prompt_ids: torch.Tensor, n: int, canvas_len: int, seed: int,
    s: GenSettings, anchors: Optional[dict[int, int]],
):
    if s.order not in ORDERS:
        raise ValueError(f"order must be one of {ORDERS}, got {s.order!r}")
    model, dev = loaded.model, loaded.device
    mask_id = model.config.mask_token_id
    temperature = 0.0 if s.temperature is None else float(s.temperature)
    P = prompt_ids.shape[1]

    x = torch.full((n, P + canvas_len), mask_id, dtype=torch.long, device=dev)
    x[:, :P] = prompt_ids.to(dev)
    for pos, tok in (anchors or {}).items():
        x[:, P + pos] = tok

    n_mask = int((x[0, P:] == mask_id).sum())
    steps = min(s.steps or n_mask, n_mask)
    base, rem = divmod(n_mask, steps) if steps else (0, 0)
    per_step = [base + (1 if i < rem else 0) for i in range(steps)]

    gen = torch.Generator(device=dev)
    gen.manual_seed(seed)
    rows = torch.arange(n, device=dev).unsqueeze(1)
    commit = torch.full((n, canvas_len), -1, dtype=torch.long, device=dev)
    positions = torch.arange(canvas_len, device=dev, dtype=torch.float64).expand(n, -1)
    neg_inf = torch.tensor(float("-inf"), dtype=torch.float64, device=dev)

    with _head_on_canvas_only(model, P):
        for i in range(steps):
            logits = model(x).logits.to(torch.float64)            # (n, canvas, vocab)
            canvas_mask = x[:, P:] == mask_id
            if temperature > 0:
                # Gumbel-max in float64, as in the reference sampler.
                noise = torch.rand(logits.shape, dtype=torch.float64, device=dev, generator=gen)
                x0 = torch.argmax(logits.exp() / (-torch.log(noise)) ** temperature, dim=-1)
            else:
                x0 = torch.argmax(logits, dim=-1)

            if s.order == "confidence":
                p = F.softmax(logits, dim=-1)
                conf = torch.gather(p, -1, x0.unsqueeze(-1)).squeeze(-1)
            elif s.order == "random":
                conf = torch.rand((n, canvas_len), dtype=torch.float64, device=dev, generator=gen)
            elif s.order == "l2r":
                conf = -positions
            else:  # r2l
                conf = positions.clone()
            conf = torch.where(canvas_mask, conf, neg_inf)

            idx = torch.topk(conf, per_step[i], dim=-1).indices   # (n, k)
            x[rows, P + idx] = x0[rows, idx]
            commit[rows, idx] = i
            del logits

    ends = end_token_ids(loaded)
    samples = []
    for row, steps_row in zip(x[:, P:].tolist(), commit.tolist()):
        sample = _decode(loaded, row, ends, mask_id)
        sample.commit_step = steps_row
        samples.append(sample)
    effective = {
        "sampler": "llada reference port (Gumbel-max float64, low-confidence remasking)",
        "temperature": temperature,
        "order": s.order,
        "steps": steps,
        "canvas_tokens": canvas_len,
        "tokens_per_step": per_step,
        "cfg_scale": 0.0,
    }
    return samples, effective, steps


# ----------------------------------------------------------------------------
# Dream
# ----------------------------------------------------------------------------

@torch.no_grad()
def _dream_sample(
    loaded: Loaded, prompt_ids: torch.Tensor, n: int, canvas_len: int, seed: int,
    s: GenSettings, anchors: Optional[dict[int, int]],
):
    model, dev = loaded.model, loaded.device
    mask_id = model.config.mask_token_id
    temperature = 0.2 if s.temperature is None else float(s.temperature)
    top_p = 0.95 if s.top_p is None else float(s.top_p)
    steps = s.steps or canvas_len
    P = prompt_ids.shape[1]

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    ids = prompt_ids.to(dev).repeat(n, 1)

    def fix_anchors(step, x, logits):
        for pos, tok in (anchors or {}).items():
            x[:, P + pos] = tok
        return x

    out = model.diffusion_generate(
        ids,
        attention_mask=torch.ones_like(ids),
        max_new_tokens=canvas_len,
        output_history=True,
        return_dict_in_generate=True,
        steps=steps,
        temperature=temperature,
        top_p=top_p,
        alg=s.dream_alg,
        alg_temp=s.dream_alg_temp,
        generation_tokens_hook_func=fix_anchors,
    )
    final = out.sequences[:, P:]
    # Step at which each position first stopped being a mask.
    commit = torch.full(final.shape, -1, dtype=torch.long, device=final.device)
    for i, snapshot in enumerate(out.history):
        newly = (snapshot[:, P:] != mask_id) & (commit == -1)
        commit[newly] = i
    for pos in (anchors or {}):
        commit[:, pos] = -1

    ends = end_token_ids(loaded)
    samples = []
    for row, steps_row in zip(final.tolist(), commit.tolist()):
        sample = _decode(loaded, row, ends, mask_id)
        sample.commit_step = steps_row
        samples.append(sample)
    effective = {
        "sampler": "dream diffusion_generate (authors' code)",
        "temperature": temperature,
        "top_p": top_p,
        "alg": s.dream_alg,
        "alg_temp": s.dream_alg_temp,
        "steps": steps,
        "canvas_tokens": canvas_len,
    }
    return samples, effective, steps


# ----------------------------------------------------------------------------
# One entry point
# ----------------------------------------------------------------------------

def generate(
    loaded: Loaded,
    user_message: str,
    *,
    n_samples: int,
    n_chars: int,
    seed: int,
    settings: Optional[GenSettings] = None,
    anchors: Optional[dict[int, int]] = None,
) -> GenOutput:
    """Draw `n_samples` replies to one user message.

    `n_chars` is the length the reply should have, in characters; the diffusion
    canvas is `n_chars + settings.canvas_slack` tokens. `anchors` maps canvas
    positions to token ids that are fixed before denoising (diffusion only).
    """
    s = settings or GenSettings()
    kind = loaded.spec.kind
    prompt_ids = chat_ids(loaded, user_message)
    if torch.cuda.is_available():
        # Hand cached blocks back first: with prompts of varying length the
        # allocator otherwise keeps growing until it spills past the 16 GB card.
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()

    if kind == "ar" and anchors:
        raise ValueError("anchors need a diffusion model")
    if kind not in ("ar", "llada", "dream"):
        raise ValueError(f"unknown model kind {kind!r}")

    per_batch = max(1, s.samples_per_batch)
    sizes = [min(per_batch, n_samples - i) for i in range(0, n_samples, per_batch)]
    seeds = [seed if j == 0 else item_seed(seed, f"batch{j}") for j in range(len(sizes))]
    samples: list[Sample] = []
    n_forward = 0
    effective: dict = {}
    for size, batch_seed in zip(sizes, seeds):
        if kind == "ar":
            part, effective, nf = _ar_sample(loaded, prompt_ids, size, batch_seed, s)
        elif kind == "llada":
            part, effective, nf = _llada_sample(
                loaded, prompt_ids, size, n_chars + s.canvas_slack, batch_seed, s, anchors)
        else:
            part, effective, nf = _dream_sample(
                loaded, prompt_ids, size, n_chars + s.canvas_slack, batch_seed, s, anchors)
        samples.extend(part)
        n_forward += nf
    effective["batch_sizes"] = sizes
    effective["batch_seeds"] = seeds

    seconds = time.time() - t0
    peak = round(torch.cuda.max_memory_allocated() / 2**30, 2) if torch.cuda.is_available() else None
    reserved = round(torch.cuda.max_memory_reserved() / 2**30, 2) if torch.cuda.is_available() else None
    return GenOutput(
        samples=samples,
        settings={"requested": asdict(s), "effective": effective, "seed": seed, "n_samples": n_samples,
                  "n_chars": n_chars},
        seconds=seconds,
        n_forward=n_forward,
        prompt_tokens=int(prompt_ids.shape[1]),
        peak_gpu_gib=peak,
        peak_reserved_gib=reserved,
    )
