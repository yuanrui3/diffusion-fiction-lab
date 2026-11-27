"""Tables for the tone probes (explore/probe_tones.py).

Usage:
    python explore/report_probes.py      # writes explore/out/probes.md and probes.json
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dfl.text import hanzi_only  # noqa: E402
from dfl.tones import PING, ZE, modern_classes  # noqa: E402

OUT = ROOT / "explore" / "out"
NAMES = {"qwen": "Qwen", "llada": "LLaDA", "dream": "Dream"}
SEED = 20261125


def boot(values_by_item, fn, n_boot=2000, seed=SEED):
    ids = sorted(values_by_item)
    rng = np.random.default_rng(seed)
    est = fn([values_by_item[i] for i in ids])
    draws = []
    for _ in range(n_boot):
        k = rng.integers(0, len(ids), len(ids))
        draws.append(fn([values_by_item[ids[j]] for j in k]))
    return est, float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def f(e, scale=100):
    return f"{scale * e[0]:+.0f} [{scale * e[1]:+.0f}, {scale * e[2]:+.0f}]"


def main() -> int:
    report = ["# Tone probes\n",
              "Post hoc; see `PROTOCOL.md` entry 36 and `explore/probe_tones.py`. Read from the models' output distributions in one forward pass; nothing is sampled.\n"]
    data = {}

    # ------------------------------------------------------------ K
    report.append("## Probe K: does the model know a character's Mandarin tone?\n")
    report.append("Question: 汉字“X”在普通话里读第几声？只回答一个数字。 The first position of the reply, restricted to the tokens 1–4. "
                  "Characters: the final characters of every original line, with one unambiguous reading. "
                  "*Tone*: the most probable digit is the right tone (chance 25%; always answering the most common tone scores the column *Majority*). "
                  "*平/仄*: the mass on 1+2 against 3+4 picks the right class (chance 50%). *On digits*: share of the whole distribution that falls on the four digits.\n")
    report.append("| Model | Characters | Tone | Majority | 平/仄 | Mean P(right tone) | On digits |")
    report.append("|---|---|---|---|---|---|---|")
    for key in ("qwen", "llada", "dream"):
        p = OUT / f"probe_K_{key}.jsonl"
        if not p.exists():
            continue
        rows = [json.loads(l) for l in open(p, encoding="utf-8")]
        tone_acc = np.mean([int(np.argmax(r["p"])) + 1 == r["tone"] for r in rows])
        maj_tone = Counter(r["tone"] for r in rows).most_common(1)[0]
        pz = np.mean([(r["p"][0] + r["p"][1] > 0.5) == (r["tone"] <= 2) for r in rows])
        pr = np.mean([r["p"][r["tone"] - 1] for r in rows])
        dm = np.mean([r["digit_mass"] for r in rows])
        pred = Counter(int(np.argmax(r["p"])) + 1 for r in rows)
        data[f"K|{key}"] = {"n": len(rows), "tone_acc": float(tone_acc), "majority": maj_tone[1] / len(rows), "pingze_acc": float(pz),
                            "p_right": float(pr), "digit_mass": float(dm), "predicted": dict(pred)}
        report.append(f"| {NAMES[key]} | {len(rows)} | {100 * tone_acc:.0f}% | {100 * maj_tone[1] / len(rows):.0f}% (tone {maj_tone[0]}) | {100 * pz:.0f}% | {pr:.2f} | {100 * dm:.0f}% |")
    report.append("")

    report.append("### K2: the same question, reply started with 第 (`explore/probe_checks.py`)\n")
    report.append("The next position is read over the digits and the Chinese numerals 一二三四 (and 一声 where it is one token), summed per tone. "
                  "*On numerals*: share of the whole distribution on those candidates.\n")
    report.append("| Model | Tone | 平/仄 | On numerals |")
    report.append("|---|---|---|---|")
    for key in ("qwen", "llada", "dream"):
        p = OUT / f"probe_K2_{key}.jsonl"
        if not p.exists():
            continue
        rows = [json.loads(l) for l in open(p, encoding="utf-8")]
        tone_acc = np.mean([int(np.argmax(r["p"])) + 1 == r["tone"] for r in rows])
        pz = np.mean([(r["p"][0] + r["p"][1] > 0.5) == (r["tone"] <= 2) for r in rows])
        nm = np.mean([r["numeral_mass"] for r in rows])
        data[f"K2|{key}"] = {"tone_acc": float(tone_acc), "pingze_acc": float(pz), "numeral_mass": float(nm)}
        report.append(f"| {NAMES[key]} | {100 * tone_acc:.0f}% | {100 * pz:.0f}% | {100 * nm:.0f}% |")
    report.append("")

    # ------------------------------------------------------------ S
    report.append("## Probe S: the last character's tone, everything else in place\n")
    report.append("The exact phase 1 prompt (bare), then the original line minus its last character; the slot that follows is read as a distribution over "
                  "tokens that are one unambiguous 汉字, and the share of that mass on 平 is recorded. *Rule*: the prompt as used, whose rule 三 states the end tone. "
                  "*No rule*: the same prompt with rule 三 removed. Forward = writing a second line (the rule wants 平), reverse = a first line (the rule wants 仄). "
                  "All 723 items of the unseen, recall and dev sets. Intervals: 95% bootstrap over items.\n")
    report.append("| Model | Prompt | P(平), writing second lines | P(平), writing first lines | Rule effect at the slot | P(original character) |")
    report.append("|---|---|---|---|---|---|")
    s_all = {}
    for key in ("qwen", "llada", "dream"):
        p = OUT / f"probe_S_{key}.jsonl"
        if not p.exists():
            continue
        rows = [json.loads(l) for l in open(p, encoding="utf-8")]
        s_all[key] = rows
        for variant in ("rule", "norule"):
            per = defaultdict(dict)
            for r in rows:
                per[r["item_id"]][r["direction"]] = r[variant]["p_ping"]
            per = {i: v for i, v in per.items() if "forward" in v and "reverse" in v}
            fwd = np.mean([v["forward"] for v in per.values()])
            rev = np.mean([v["reverse"] for v in per.values()])
            eff = boot(per, lambda vs: np.mean([v["forward"] - v["reverse"] for v in vs]))
            po = np.mean([r[variant]["p_original"] for r in rows if r[variant]["p_original"] is not None])
            data[f"S|{key}|{variant}"] = {"fwd": float(fwd), "rev": float(rev), "rule_effect": eff, "p_original": float(po)}
            report.append(f"| {NAMES[key]} | {'rule' if variant == 'rule' else 'no rule'} | {100 * fwd:.0f}% | {100 * rev:.0f}% | {f(eff)} | {po:.2f} |")
    report.append("")

    report.append("### The same with the rest of the canvas masked (`explore/probe_S_masked_tail.py`)\n")
    report.append("| Model | Prompt | P(平), second lines | P(平), first lines | P(original character) | Mass on single 汉字 |")
    report.append("|---|---|---|---|---|---|")
    for key in ("llada", "dream"):
        p = OUT / f"probe_S_{key}_maskedtail.jsonl"
        if not p.exists():
            continue
        rows = [json.loads(l) for l in open(p, encoding="utf-8")]
        for variant in ("rule", "norule"):
            fw = np.mean([r[variant]["p_ping"] for r in rows if r["direction"] == "forward"])
            rv = np.mean([r[variant]["p_ping"] for r in rows if r["direction"] == "reverse"])
            po = np.mean([r[variant]["p_original"] for r in rows if r[variant]["p_original"] is not None])
            hm = np.mean([r[variant]["hanzi_mass"] for r in rows])
            data[f"S_masked|{key}|{variant}"] = {"fwd": float(fw), "rev": float(rv), "p_original": float(po), "hanzi_mass": float(hm)}
            report.append(f"| {NAMES[key]} | {'rule' if variant == 'rule' else 'no rule'} | {100 * fw:.0f}% | {100 * rv:.0f}% | {po:.3f} | {100 * hm:.0f}% |")
    report.append("")

    report.append("### What stating the rule does to the slot\n")
    report.append("P(平) with the rule minus P(平) without it, paired by item. A model that obeys the rule should move toward 平 when writing second lines (positive) "
                  "and toward 仄 when writing first lines (negative).\n")
    report.append("| Model | Writing second lines | Writing first lines |")
    report.append("|---|---|---|")
    for key, rows in s_all.items():
        cells = []
        for d in ("forward", "reverse"):
            per = {r["item_id"]: r["rule"]["p_ping"] - r["norule"]["p_ping"] for r in rows if r["direction"] == d}
            e = boot(per, lambda vs: float(np.mean(vs)))
            data[f"S|{key}|instruction|{d}"] = e
            cells.append(f(e))
        report.append(f"| {NAMES[key]} | {cells[0]} | {cells[1]} |")
    report.append("")

    report.append("### Rule or imitation at the slot (recall set, rule prompt)\n")
    report.append("As in `existing_data.md`, but for the slot distribution: P(平) by direction and by the given line's final tone (modern).\n")
    report.append("| Model | Fwd, given 仄 | Fwd, given 平 | Rev, given 仄 | Rev, given 平 | Rule effect | Imitation effect |")
    report.append("|---|---|---|---|---|---|---|")
    for key, rows in s_all.items():
        cells = defaultdict(list)
        for r in rows:
            if r["set"] != "recall":
                continue
            g = modern_classes(hanzi_only(r["given"])[-1])[0]
            if g not in (PING, ZE):
                continue
            cells[(r["direction"], g)].append(r["rule"]["p_ping"])
        m = {k: float(np.mean(v)) for k, v in cells.items()}
        rule = 0.5 * ((m[("forward", PING)] - m[("reverse", PING)]) + (m[("forward", ZE)] - m[("reverse", ZE)]))
        imit = 0.5 * ((m[("forward", PING)] - m[("forward", ZE)]) + (m[("reverse", PING)] - m[("reverse", ZE)]))
        data[f"S|{key}|recall_cells"] = {f"{d}|{g}": v for (d, g), v in m.items()} | {"rule": rule, "imit": imit}
        report.append(f"| {NAMES[key]} | {100 * m[('forward', ZE)]:.0f}% | {100 * m[('forward', PING)]:.0f}% | {100 * m[('reverse', ZE)]:.0f}% | {100 * m[('reverse', PING)]:.0f}% | {100 * rule:+.0f} | {100 * imit:+.0f} |")
    report.append("")

    (OUT / "probes.md").write_text("\n".join(report) + "\n", encoding="utf-8", newline="\n")
    (OUT / "probes.json").write_text(json.dumps(data, ensure_ascii=False, indent=1, default=float) + "\n", encoding="utf-8", newline="\n")
    print("\n".join(report))
    return 0


if __name__ == "__main__":
    sys.exit(main())
