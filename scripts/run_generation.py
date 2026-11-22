"""Generate couplet lines with one model and append them to a run file.

Usage:
    python scripts/run_generation.py --model qwen --items data/dev/dev_pool.jsonl \
        --ids jhy-005,jhy-010 --condition bare --out runs/smoke/qwen.jsonl

One model per process (one model on the GPU at a time). The run file is JSONL:
a header line describing the model, precision, code and settings, then one line
per (item, direction, condition) holding the prompt, the seed and every sample.
Re-running with the same --out skips what is already there, so an interrupted
run can be continued.

Conditions:
    summary  the prompt includes the item's summary (items without one are skipped)
    bare     the prompt gives the line alone
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

from dfl.generate import GenSettings, generate, item_seed
from dfl.models import load
from dfl.prompts import FORWARD, PROMPT_VERSION, REVERSE, build_prompt
from dfl.text import count_hanzi

ROOT = Path(__file__).resolve().parents[1]


def code_version() -> str:
    try:
        head = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True)
        dirty = subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain"], capture_output=True, text=True)
        if head.returncode == 0:
            return head.stdout.strip() + ("+dirty" if dirty.stdout.strip() else "")
    except OSError:
        pass
    return "uncommitted"


def read_items(path: Path, ids: list[str] | None, limit: int | None) -> list[dict]:
    items = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if ids:
        by_id = {it["id"]: it for it in items}
        missing = [i for i in ids if i not in by_id]
        if missing:
            raise SystemExit(f"unknown item ids: {missing}")
        items = [by_id[i] for i in ids]
    if limit:
        items = items[:limit]
    return items


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["qwen", "dream", "llada"])
    ap.add_argument("--label", default=None,
                    help="name of this arm (a model at one sampling setting); defaults to the model key")
    ap.add_argument("--items", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--ids", default=None, help="comma-separated item ids")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--condition", choices=["summary", "bare"], required=True)
    ap.add_argument("--directions", default="forward,reverse")
    ap.add_argument("--n-samples", type=int, default=8)
    ap.add_argument("--base-seed", type=int, default=20261119)
    ap.add_argument("--precision", default="bf16", choices=["bf16", "int8"])
    ap.add_argument("--temperature", type=float, default=None)
    ap.add_argument("--top-p", type=float, default=None)
    ap.add_argument("--canvas-slack", type=int, default=4)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--order", default="confidence")
    ap.add_argument("--samples-per-batch", type=int, default=4)
    ap.add_argument("--tag", default="", help="free label stored in the header, e.g. smoke, pilot, full")
    ap.add_argument("--prompt-version", default=PROMPT_VERSION, choices=["v1", "v2"])
    args = ap.parse_args()
    label = args.label or args.model

    settings = GenSettings(
        canvas_slack=args.canvas_slack, steps=args.steps, temperature=args.temperature,
        top_p=args.top_p, order=args.order, samples_per_batch=args.samples_per_batch,
    )
    items = read_items(args.items, args.ids.split(",") if args.ids else None, args.limit)
    directions = [d for d in args.directions.split(",") if d]
    assert all(d in (FORWARD, REVERSE) for d in directions)

    done: set[tuple[str, str, str]] = set()
    if args.out.exists():
        for line in args.out.read_text(encoding="utf-8").splitlines():
            rec = json.loads(line)
            if rec.get("type") == "item":
                done.add((rec["item_id"], rec["direction"], rec["condition"]))

    todo = []
    for it in items:
        if args.condition == "summary" and not it.get("summary"):
            continue
        for d in directions:
            if (it["id"], d, args.condition) not in done:
                todo.append((it, d))
    print(f"{label}: {len(todo)} to generate, {len(done)} already in {args.out}", flush=True)
    if not todo:
        return 0

    loaded = load(args.model, precision=args.precision)
    print(f"loaded {loaded.spec.repo} in {loaded.load_seconds:.0f} s, "
          f"{loaded.info['gpu_gib_after_load']} GiB on GPU, precision {loaded.precision}, "
          f"embedding on CPU: {loaded.embed_on_cpu}", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "a", encoding="utf-8", newline="\n") as fh:
        header = {
            "type": "header",
            "tag": args.tag,
            "label": label,
            "started": time.strftime("%Y-%m-%d %H:%M:%S"),
            "model": loaded.describe(),
            "load_seconds": round(loaded.load_seconds, 1),
            "prompt_version": args.prompt_version,
            "code_version": code_version(),
            "base_seed": args.base_seed,
            "n_samples": args.n_samples,
            "condition": args.condition,
            "settings_requested": asdict(settings),
            "items_file": str(args.items.as_posix()),
        }
        fh.write(json.dumps(header, ensure_ascii=False) + "\n")
        fh.flush()

        t_start = time.time()
        for k, (it, direction) in enumerate(todo, 1):
            given = it["first"] if direction == FORWARD else it["second"]
            reference = it["second"] if direction == FORWARD else it["first"]
            n_chars = count_hanzi(given)
            summary = it.get("summary") if args.condition == "summary" else None
            prompt = build_prompt(given, direction, n_chars, summary, version=args.prompt_version)
            seed = item_seed(args.base_seed, args.model, it["id"], direction, args.condition)
            out = generate(loaded, prompt, n_samples=args.n_samples, n_chars=n_chars, seed=seed, settings=settings)
            rec = {
                "type": "item",
                "model": args.model,
                "label": label,
                "item_id": it["id"],
                "set": it.get("set"),
                "direction": direction,
                "condition": args.condition,
                "prompt_version": args.prompt_version,
                "given": given,
                "reference": reference,
                "n_chars": n_chars,
                "prompt": prompt,
                "seed": seed,
                "settings": out.settings["effective"],
                "prompt_tokens": out.prompt_tokens,
                "n_forward": out.n_forward,
                "seconds": round(out.seconds, 3),
                "peak_gpu_gib": out.peak_gpu_gib,
                "peak_reserved_gib": out.peak_reserved_gib,
                "samples": [asdict(smp) for smp in out.samples],
            }
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fh.flush()
            if k <= 3 or k % 20 == 0 or k == len(todo):
                elapsed = time.time() - t_start
                eta = elapsed / k * (len(todo) - k)
                print(f"  [{k}/{len(todo)}] {it['id']} {direction}: {out.seconds:.1f} s, "
                      f"peak {out.peak_gpu_gib}/{out.peak_reserved_gib} GiB, eta {eta / 60:.1f} min | "
                      f"{' / '.join(smp.raw.strip()[:16] for smp in out.samples[:3])}", flush=True)
    print(f"done: {len(todo)} records in {(time.time() - t_start) / 60:.1f} min", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
