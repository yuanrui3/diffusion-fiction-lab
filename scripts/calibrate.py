"""Pick the diffusion sampling temperature on the dev calibration items.

Usage:
    python scripts/calibrate.py run       # generate (about 25 minutes of GPU time)
    python scripts/calibrate.py report    # score and choose

Rule fixed in PROTOCOL.md (entry 8) before any of this was run: Qwen runs at its
vendor defaults; for each diffusion model the temperature is the value from
{0.3, 0.5, 0.7, 1.0} whose mean number of distinct answers per 8 samples is
closest to Qwen's. A tie goes to the lower temperature.

Items: the 20 dev items in the "calibration" split, both directions, no summary.

Second step, canvas slack:
    python scripts/calibrate.py slack-run      # the chosen temperatures at slack 6 and 8
    python scripts/calibrate.py slack-report

Rule, written down before the slack runs were made: one slack for both
diffusion models, the value from {4, 6, 8} with the highest all-rules pass rate
averaged over the two models at their chosen temperatures. A tie goes to the
smaller canvas.
"""

from __future__ import annotations

import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs" / "calib"
GRID = [0.3, 0.5, 0.7, 1.0]
CONFIGS = [("qwen", None)] + [(m, t) for m in ("dream", "llada") for t in GRID]


def run_file(model: str, temperature) -> Path:
    return OUT / (f"{model}_default.jsonl" if temperature is None else f"{model}_T{temperature}.jsonl")


def run() -> int:
    ids = json.loads((ROOT / "data" / "dev" / "splits.json").read_text(encoding="utf-8"))["calibration"]
    for model, temperature in CONFIGS:
        cmd = [sys.executable, str(ROOT / "scripts" / "run_generation.py"), "--model", model,
               "--items", str(ROOT / "data" / "dev" / "dev.jsonl"), "--ids", ",".join(ids),
               "--condition", "bare", "--out", str(run_file(model, temperature)), "--tag", "calibration"]
        if temperature is not None:
            cmd += ["--temperature", str(temperature)]
        print(f"=== {model} temperature {temperature}", flush=True)
        code = subprocess.run(cmd, cwd=ROOT).returncode
        if code != 0:
            print(f"run failed with exit code {code}", flush=True)
            return code
    print("CALIBRATION RUNS COMPLETE", flush=True)
    return 0


SLACKS = [4, 6, 8]


def slack_file(model: str, temperature: float, slack: int) -> Path:
    return run_file(model, temperature) if slack == 4 else OUT / f"{model}_T{temperature}_slack{slack}.jsonl"


def slack_run() -> int:
    ids = json.loads((ROOT / "data" / "dev" / "splits.json").read_text(encoding="utf-8"))["calibration"]
    chosen = json.loads((OUT / "chosen.json").read_text(encoding="utf-8"))["chosen"]
    for model, temperature in chosen.items():
        for slack in SLACKS[1:]:
            cmd = [sys.executable, str(ROOT / "scripts" / "run_generation.py"), "--model", model,
                   "--items", str(ROOT / "data" / "dev" / "dev.jsonl"), "--ids", ",".join(ids),
                   "--condition", "bare", "--temperature", str(temperature), "--canvas-slack", str(slack),
                   "--out", str(slack_file(model, temperature, slack)), "--tag", "calibration-slack"]
            print(f"=== {model} temperature {temperature} slack {slack}", flush=True)
            code = subprocess.run(cmd, cwd=ROOT).returncode
            if code != 0:
                return code
    print("SLACK RUNS COMPLETE", flush=True)
    return 0


def summarise(path: Path, pingshui, tagger, coarse) -> dict:
    from dfl.scoring import score_record

    rows, seconds = [], []
    for line in path.read_text(encoding="utf-8").splitlines():
        rec = json.loads(line)
        if rec.get("type") == "item":
            rows.extend(score_record(rec, pingshui=pingshui, tagger=tagger, coarse=coarse))
            seconds.append(rec["seconds"])
    groups = defaultdict(list)
    for r in rows:
        groups[(r["item_id"], r["direction"])].append(r)
    rate = lambda key: sum(bool(r[key]) for r in rows) / len(rows)
    pos = [r["pos_match"] for r in rows if r["pos_match"] is not None]
    return {
        "records": len(groups),
        "distinct": sum(len({r["answer"] for r in g}) for g in groups.values()) / len(groups),
        "length": rate("rule_length"), "tone": rate("rule_tone_modern"),
        "no_repeat": rate("rule_no_repeat"), "all": rate("all_rules_modern"),
        "pos": sum(pos) / len(pos) if pos else float("nan"),
        "sec": sum(seconds) / len(seconds),
    }


def slack_report() -> int:
    from dfl.scoring import default_tagger
    from dfl.tones import Pingshui

    pingshui = Pingshui.load()
    tagger, coarse = default_tagger()
    chosen = json.loads((OUT / "chosen.json").read_text(encoding="utf-8"))
    print(f"{'model':6s} {'temp':5s} {'slack':>5s} {'distinct/8':>10s} {'length':>7s} {'tone':>6s} {'no rep':>7s} {'all':>6s} {'pos':>6s} {'s/rec':>6s}")
    mean_all = {}
    for slack in SLACKS:
        vals = []
        for model, temperature in chosen["chosen"].items():
            path = slack_file(model, temperature, slack)
            if not path.exists():
                print(f"missing {path.name}")
                continue
            st = summarise(path, pingshui, tagger, coarse)
            vals.append(st["all"])
            print(f"{model:6s} {temperature:<5} {slack:5d} {st['distinct']:10.2f} {st['length']:7.0%} {st['tone']:6.0%} "
                  f"{st['no_repeat']:7.0%} {st['all']:6.0%} {st['pos']:6.2f} {st['sec']:6.1f}")
        if len(vals) == len(chosen["chosen"]):
            mean_all[slack] = sum(vals) / len(vals)
    if mean_all:
        best = max(mean_all, key=lambda k: (round(mean_all[k], 4), -k))
        print()
        print("mean all-rules rate over the two diffusion models: " + ", ".join(f"slack {k}: {v:.1%}" for k, v in mean_all.items()))
        print(f"chosen canvas slack: {best}")
        chosen["canvas_slack"] = best
        chosen["slack_mean_all_rules"] = {str(k): v for k, v in mean_all.items()}
        (OUT / "chosen.json").write_text(json.dumps(chosen, indent=1), encoding="utf-8")
    return 0


def report() -> int:
    from dfl.scoring import default_tagger, score_record
    from dfl.tones import Pingshui

    pingshui = Pingshui.load()
    tagger, coarse = default_tagger()
    stats = {}
    for model, temperature in CONFIGS:
        path = run_file(model, temperature)
        if not path.exists():
            print(f"missing {path.name}")
            continue
        rows, seconds = [], []
        for line in path.read_text(encoding="utf-8").splitlines():
            rec = json.loads(line)
            if rec.get("type") == "item":
                rows.extend(score_record(rec, pingshui=pingshui, tagger=tagger, coarse=coarse))
                seconds.append(rec["seconds"])
        groups = defaultdict(list)
        for r in rows:
            groups[(r["item_id"], r["direction"])].append(r)
        rate = lambda key: sum(bool(r[key]) for r in rows) / len(rows)
        pos = [r["pos_match"] for r in rows if r["pos_match"] is not None]
        stats[(model, temperature)] = {
            "records": len(groups),
            "distinct": sum(len({r["answer"] for r in g}) for g in groups.values()) / len(groups),
            "length": rate("rule_length"), "tone": rate("rule_tone_modern"),
            "no_repeat": rate("rule_no_repeat"), "all": rate("all_rules_modern"),
            "pos": sum(pos) / len(pos) if pos else float("nan"),
            "sec": sum(seconds) / len(seconds),
        }

    print(f"{'model':6s} {'temp':8s} {'records':>7s} {'distinct/8':>10s} {'length':>7s} {'tone':>6s} {'no rep':>7s} {'all':>6s} {'pos':>6s} {'s/rec':>6s}")
    for (model, temperature), s in stats.items():
        t = "default" if temperature is None else str(temperature)
        print(f"{model:6s} {t:8s} {s['records']:7d} {s['distinct']:10.2f} {s['length']:7.0%} {s['tone']:6.0%} "
              f"{s['no_repeat']:7.0%} {s['all']:6.0%} {s['pos']:6.2f} {s['sec']:6.1f}")

    if ("qwen", None) not in stats:
        return 1
    target = stats[("qwen", None)]["distinct"]
    print(f"\ntarget: Qwen at its defaults gives {target:.2f} distinct answers per 8")
    chosen = {}
    for model in ("dream", "llada"):
        have = [(abs(stats[(model, t)]["distinct"] - target), t) for t in GRID if (model, t) in stats]
        if have:
            chosen[model] = min(have)[1]
            print(f"chosen for {model}: temperature {chosen[model]} "
                  f"({stats[(model, chosen[model])]['distinct']:.2f} distinct per 8)")
    (OUT / "chosen.json").write_text(json.dumps({"target_distinct": target, "chosen": chosen}, indent=1) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    action = sys.argv[1] if len(sys.argv) > 1 else ""
    actions = {"run": run, "report": report, "slack-run": slack_run, "slack-report": slack_report}
    if action not in actions:
        print(__doc__)
        sys.exit(2)
    sys.exit(actions[action]())
