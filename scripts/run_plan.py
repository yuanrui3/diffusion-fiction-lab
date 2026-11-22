"""Run a whole plan (pilot or full) from the frozen configuration.

Usage:
    python scripts/run_plan.py pilot
    python scripts/run_plan.py full [--arms qwen,dream,llada]

Reads `config/phase1.json` and calls `run_generation.py` once per arm and part,
one process at a time. An arm is a model at one sampling setting; it has a
label (for example "dream" for the matched-diversity arm and "dream-default"
for the vendor-default arm). Run files go to `runs/<plan>/<label>_<part>.jsonl`;
an interrupted plan can be restarted and continues where it stopped.

Plans:
    pilot   the 10 dev pilot items, with summary (8 of them have one) and bare
    full    unseen with summary, unseen bare, recall with summary (48), recall bare (580)
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config" / "phase1.json"


def parts_for(plan: str) -> list[dict]:
    if plan == "pilot":
        ids = json.loads((ROOT / "data" / "dev" / "splits.json").read_text(encoding="utf-8"))["pilot"]
        return [
            {"name": "dev_summary", "items": "data/dev/dev.jsonl", "condition": "summary", "ids": ids},
            {"name": "dev_bare", "items": "data/dev/dev.jsonl", "condition": "bare", "ids": ids},
        ]
    if plan == "full":
        return [
            {"name": "unseen_summary", "items": "data/unseen/huohou.jsonl", "condition": "summary"},
            {"name": "unseen_bare", "items": "data/unseen/huohou.jsonl", "condition": "bare"},
            {"name": "recall_summary", "items": "data/recall/recall.jsonl", "condition": "summary"},
            {"name": "recall_bare", "items": "data/recall/recall.jsonl", "condition": "bare"},
        ]
    raise SystemExit(f"unknown plan {plan!r}")


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        return 2
    plan = args[0]
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    arms = cfg["arms"]
    if "--arms" in args:
        wanted = args[args.index("--arms") + 1].split(",")
        unknown = [w for w in wanted if w not in {a["label"] for a in arms}]
        if unknown:
            raise SystemExit(f"unknown arms: {unknown}")
        arms = [a for a in arms if a["label"] in wanted]

    t0 = time.time()
    for arm in arms:
        for part in parts_for(plan):
            out = ROOT / "runs" / plan / f"{arm['label']}_{part['name']}.jsonl"
            cmd = [
                sys.executable, str(ROOT / "scripts" / "run_generation.py"),
                "--model", arm["model"], "--label", arm["label"],
                "--items", str(ROOT / part["items"]), "--condition", part["condition"],
                "--out", str(out), "--tag", plan,
                "--n-samples", str(arm["n_samples"]), "--base-seed", str(cfg["base_seed"]),
                "--canvas-slack", str(cfg["canvas_slack"]), "--samples-per-batch", str(cfg["samples_per_batch"]),
                "--precision", cfg["precision"], "--prompt-version", cfg["prompt_version"],
            ]
            if part.get("ids"):
                cmd += ["--ids", ",".join(part["ids"])]
            if arm.get("temperature") is not None:
                cmd += ["--temperature", str(arm["temperature"])]
            print(f"=== {plan}: {arm['label']} / {part['name']}  ({(time.time() - t0) / 60:.0f} min elapsed)", flush=True)
            code = subprocess.run(cmd, cwd=ROOT).returncode
            if code != 0:
                print(f"FAILED with exit code {code}: {arm['label']} / {part['name']}", flush=True)
                return code
    print(f"PLAN COMPLETE: {plan} in {(time.time() - t0) / 60:.0f} min", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
