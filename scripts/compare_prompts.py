"""Compare the two prompt layouts on dev items, for the three primary arms.

Usage:
    python scripts/compare_prompts.py run       # generate with layout v2 (about 12 minutes)
    python scripts/compare_prompts.py report    # score both layouts and apply the rule

Items: the 10 pilot items (with summary where there is one, and bare) and the
20 calibration items (bare). The v1 side reuses runs already made at the frozen
sampling settings: the pilot, and the calibration runs at slack 6.

Rule, written in PROTOCOL.md before the v2 runs were made: adopt v2 only if its
all-rules pass rate, averaged over the three primary arms (each arm's rate is
over all its samples on these items), is higher than v1's. Otherwise keep v1.
"""

from __future__ import annotations

import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs" / "prompt_v2"
ARMS = ["qwen", "dream", "llada"]


def config() -> dict:
    return json.loads((ROOT / "config" / "phase1.json").read_text(encoding="utf-8"))


def parts() -> list[dict]:
    splits = json.loads((ROOT / "data" / "dev" / "splits.json").read_text(encoding="utf-8"))
    return [
        {"name": "pilot_summary", "ids": splits["pilot"], "condition": "summary"},
        {"name": "pilot_bare", "ids": splits["pilot"], "condition": "bare"},
        {"name": "calib_bare", "ids": splits["calibration"], "condition": "bare"},
    ]


def v1_files(arm: dict, slack: int) -> list[Path]:
    calib = ROOT / "runs" / "calib"
    if arm["temperature"] is None:
        calib_file = calib / f"{arm['model']}_default.jsonl"
    elif slack == 4:
        calib_file = calib / f"{arm['model']}_T{arm['temperature']}.jsonl"
    else:
        calib_file = calib / f"{arm['model']}_T{arm['temperature']}_slack{slack}.jsonl"
    pilot = ROOT / "runs" / "pilot"
    return [pilot / f"{arm['label']}_dev_summary.jsonl", pilot / f"{arm['label']}_dev_bare.jsonl", calib_file]


def run() -> int:
    cfg = config()
    for arm in [a for a in cfg["arms"] if a["label"] in ARMS]:
        for part in parts():
            cmd = [
                sys.executable, str(ROOT / "scripts" / "run_generation.py"),
                "--model", arm["model"], "--label", arm["label"], "--prompt-version", "v2",
                "--items", str(ROOT / "data" / "dev" / "dev.jsonl"), "--ids", ",".join(part["ids"]),
                "--condition", part["condition"], "--out", str(OUT / f"{arm['label']}_{part['name']}.jsonl"),
                "--tag", "prompt-v2", "--n-samples", str(arm["n_samples"]), "--base-seed", str(cfg["base_seed"]),
                "--canvas-slack", str(cfg["canvas_slack"]), "--samples-per-batch", str(cfg["samples_per_batch"]),
            ]
            if arm.get("temperature") is not None:
                cmd += ["--temperature", str(arm["temperature"])]
            print(f"=== v2: {arm['label']} / {part['name']}", flush=True)
            code = subprocess.run(cmd, cwd=ROOT).returncode
            if code != 0:
                return code
    print("PROMPT V2 RUNS COMPLETE", flush=True)
    return 0


def load_rows(files: list[Path], pingshui, tagger, coarse) -> list[dict]:
    from dfl.scoring import score_record

    rows = []
    for f in files:
        for line in f.read_text(encoding="utf-8").splitlines():
            rec = json.loads(line)
            if rec.get("type") == "item":
                rows.extend(score_record(rec, pingshui=pingshui, tagger=tagger, coarse=coarse))
    return rows


def report() -> int:
    from dfl.scoring import default_tagger
    from dfl.tones import Pingshui

    cfg = config()
    pingshui = Pingshui.load()
    tagger, coarse = default_tagger()
    rate = lambda rows, key: sum(bool(r[key]) for r in rows) / len(rows)

    means = {}
    print(f"{'layout':6s} {'arm':6s} {'condition':9s} {'samples':>7s} {'length':>7s} {'tone':>6s} {'no rep':>7s} {'all':>6s} {'pure':>6s} {'distinct/8':>10s}")
    for version in ("v1", "v2"):
        per_arm = []
        for arm in [a for a in cfg["arms"] if a["label"] in ARMS]:
            files = (v1_files(arm, cfg["canvas_slack"]) if version == "v1"
                     else [OUT / f"{arm['label']}_{p['name']}.jsonl" for p in parts()])
            missing = [f for f in files if not f.exists()]
            if missing:
                print(f"missing: {[m.name for m in missing]}")
                return 1
            rows = load_rows(files, pingshui, tagger, coarse)
            per_arm.append(rate(rows, "all_rules_modern"))
            for cond in ("summary", "bare"):
                g = [r for r in rows if r["condition"] == cond]
                cells = defaultdict(set)
                for r in g:
                    cells[(r["item_id"], r["direction"])].add(r["answer"])
                print(f"{version:6s} {arm['label']:6s} {cond:9s} {len(g):7d} {rate(g, 'rule_length'):7.0%} {rate(g, 'rule_tone_modern'):6.0%} "
                      f"{rate(g, 'rule_no_repeat'):7.0%} {rate(g, 'all_rules_modern'):6.0%} {rate(g, 'pure'):6.0%} "
                      f"{sum(len(v) for v in cells.values()) / len(cells):10.2f}")
        means[version] = sum(per_arm) / len(per_arm)
        print(f"   {version}: all-rules pass per arm {[f'{x:.1%}' for x in per_arm]}, mean {means[version]:.1%}\n")
    choice = "v2" if means["v2"] > means["v1"] else "v1"
    print(f"chosen prompt layout: {choice}")
    (OUT / "chosen.json").write_text(json.dumps({"means": means, "chosen": choice}, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    action = sys.argv[1] if len(sys.argv) > 1 else ""
    if action not in ("run", "report"):
        print(__doc__)
        sys.exit(2)
    sys.exit(run() if action == "run" else report())
