"""Exploratory analyses of the phase 1 outputs (no new generation).

Usage:
    python explore/analyse_existing.py            # writes explore/out/existing_data.md and .json

Post hoc, not part of the frozen plan (PROTOCOL.md entry 36). Reads only files
that already exist: results/full_scored.jsonl, runs/full/, runs/calib/ and the
item files. Four questions:

  1. Tone. Does a model choose the line-final tone by the rule (depends on the
     direction), by imitation (depends on the given line's final tone), or by a
     fixed preference (depends on neither)? On the unseen set the given line's
     tone and the direction always agree with the rule, so only the recall set,
     where the classical originals often break the convention, separates them.
  2. Length. How far off the required length are the lines, and for the
     diffusion models, does the output grow with the canvas?
  3. Order. In which order do the diffusion samplers fill the canvas
     (recorded as commit_step for every position)?
  4. Copying. How many characters of the answer are taken from the summary or
     from the given line, and does that track the relevance score?
"""

from __future__ import annotations

import glob
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dfl.text import hanzi_only  # noqa: E402
from dfl.tones import PING, ZE, Pingshui, modern_classes  # noqa: E402

OUT = ROOT / "explore" / "out"
MODELS = ["qwen", "llada", "dream", "llada-default", "dream-default"]
NAMES = {"qwen": "Qwen", "llada": "LLaDA 0.7", "dream": "Dream 0.5", "llada-default": "LLaDA 0", "dream-default": "Dream 0.2"}
END_TOKENS = {"llada": {126081, 126348}, "dream": {151643, 151645}}
RNG_SEED = 20261125
N_BOOT = 2000


def pct(x):
    return "–" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{100 * x:.0f}%"


def ci_str(est, lo, hi, scale=100, unit=""):
    if est is None:
        return "–"
    return f"{scale * est:+.0f}{unit} [{scale * lo:+.0f}, {scale * hi:+.0f}]"


_PS = None


def final_class(line: str, system: str = "modern"):
    """Tone class of the line's last 汉字: 平 or 仄, or None if unknown or (classically) both."""
    global _PS
    h = hanzi_only(line)
    if not h:
        return None
    if system == "classical":
        if _PS is None:
            _PS = Pingshui.load()
        c = _PS.klass(h[-1])
    else:
        c = modern_classes(h[-1])[0]
    return c if c in (PING, ZE) else None


def load_rows():
    rows = []
    with open(ROOT / "results" / "full_scored.jsonl", encoding="utf-8") as fh:
        for line in fh:
            rows.append(json.loads(line))
    return rows


def load_items():
    items = {}
    for f in ["data/unseen/huohou.jsonl", "data/recall/recall.jsonl", "data/dev/dev.jsonl"]:
        for line in open(ROOT / f, encoding="utf-8"):
            it = json.loads(line)
            items[it["id"]] = it
    return items


# ------------------------------------------------------------------ 1. tone

def tone_cells(rows, model, sets=("recall",), conditions=("bare", "summary"), system="modern"):
    """{item: {(direction, given_class): [is_ping, ...]}} for one model."""
    field = "end_modern_class" if system == "modern" else "end_classical_class"
    out = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if r["model"] != model or r["set"] not in sets or r["condition"] not in conditions:
            continue
        if r.get(field) not in (PING, ZE):
            continue
        g = final_class(r["given"], system)
        if g is None:
            continue
        out[r["item_id"]][(r["direction"], g)].append(1.0 if r[field] == PING else 0.0)
    return out


def reference_cells(items, ids, system="modern"):
    """The same table for the original lines: forward = the original second line, reverse = the first."""
    out = defaultdict(lambda: defaultdict(list))
    for i in ids:
        it = items[i]
        for direction, given, written in (("forward", it["first"], it["second"]), ("reverse", it["second"], it["first"])):
            g, w = final_class(given, system), final_class(written, system)
            if g is None or w is None:
                continue
            out[i][(direction, g)].append(1.0 if w == PING else 0.0)
    return out


def effects(cells):
    """Rule effect, imitation effect and overall rate of 平, from the 2x2 table pooled over items.

    rule  = P(平 | forward) - P(平 | reverse), averaged over the given-line tone
    imit  = P(平 | given ends 平) - P(平 | given ends 仄), averaged over direction
    Ideal rule follower: rule = +1, imit = 0. Pure imitator: rule = 0, imit = +1.
    Opposition (the classical habit of contrasting tones): imit < 0.
    """
    keys = [("forward", PING), ("forward", ZE), ("reverse", PING), ("reverse", ZE)]
    s = {k: 0.0 for k in keys}
    n = {k: 0 for k in keys}
    for per in cells.values():
        for k, v in per.items():
            s[k] += sum(v)
            n[k] += len(v)
    p = {k: (s[k] / n[k] if n[k] else np.nan) for k in keys}
    rule = 0.5 * ((p[("forward", PING)] - p[("reverse", PING)]) + (p[("forward", ZE)] - p[("reverse", ZE)]))
    imit = 0.5 * ((p[("forward", PING)] - p[("forward", ZE)]) + (p[("reverse", PING)] - p[("reverse", ZE)]))
    total = sum(s.values()) / max(1, sum(n.values()))
    return {"p": p, "n": n, "rule": rule, "imit": imit, "ping": total}


def boot_effects(cells, n_boot=N_BOOT, seed=RNG_SEED):
    ids = sorted(cells)
    rng = np.random.default_rng(seed)
    draws = {"rule": [], "imit": [], "ping": []}
    for _ in range(n_boot):
        pick = rng.integers(0, len(ids), len(ids))
        sub = {}
        for j, i in enumerate(pick):
            sub[(ids[i], j)] = cells[ids[i]]
        e = effects(sub)
        for k in draws:
            draws[k].append(e[k])
    est = effects(cells)
    out = {}
    for k in draws:
        d = np.array(draws[k], dtype=float)
        d = d[~np.isnan(d)]
        out[k] = (est[k], float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5)))
    out["p"], out["n"] = est["p"], est["n"]
    return out


def section_tone(rows, items, report, data):
    report.append("## 1. How the line-final tone is chosen\n")
    report.append("Modern tones (pypinyin). The rule asks a second line (forward) to end on 平 and a first line (reverse) on 仄. "
                  "On the unseen set the given line always ends the conventional way (93% of items keep both rules), so direction and "
                  "the given line's tone always point the same way there. The recall set separates them: its classical originals "
                  "often break the convention, so a given first line may end on 平 and a given second line on 仄.\n")

    # base rates
    report.append("### Share of lines ending on 平\n")
    report.append("| Lines | Second lines | First lines | All |")
    report.append("|---|---|---|---|")
    for label, prefix in (("火候 originals (unseen)", "huohou"), ("Classical originals (recall)", None), ("镜花缘 originals (dev)", "jhy")):
        sec, fir = [], []
        for it in items.values():
            if prefix == "huohou" and it.get("set") != "unseen":
                continue
            if prefix is None and it.get("set") != "recall":
                continue
            if prefix == "jhy" and it.get("set") != "dev":
                continue
            a, b = final_class(it["first"]), final_class(it["second"])
            if a:
                fir.append(a == PING)
            if b:
                sec.append(b == PING)
        report.append(f"| {label} | {pct(np.mean(sec))} (n={len(sec)}) | {pct(np.mean(fir))} (n={len(fir)}) | {pct(np.mean(sec + fir))} |")
    for m in MODELS:
        by_dir = {}
        for d in ("forward", "reverse"):
            v = [r["end_modern_class"] == PING for r in rows if r["model"] == m and r["direction"] == d and r.get("end_modern_class") in (PING, ZE)]
            by_dir[d] = v
        report.append(f"| {NAMES[m]}, all its lines | {pct(np.mean(by_dir['forward']))} (writing second lines) | {pct(np.mean(by_dir['reverse']))} (writing first lines) | {pct(np.mean(by_dir['forward'] + by_dir['reverse']))} |")
    report.append("")

    # rule vs imitation on the recall set
    report.append("### Rule or imitation? Recall set, both conditions\n")
    report.append("Share of answers ending on 平 in each cell; then the two effects with 95% bootstrap intervals over items. "
                  "*Rule effect* = how much more often the line ends on 平 when the rule asks for 平 (forward) than when it asks for 仄 (reverse), "
                  "holding the given line's tone fixed: +100 for a perfect rule follower, 0 for a model blind to the rule. "
                  "*Imitation effect* = how much more often the line ends on 平 when the given line ends on 平, holding direction fixed: "
                  "positive means copying the given tone, negative means contrasting with it.\n")
    report.append("| Writer | Fwd, given 仄 | Fwd, given 平 | Rev, given 仄 | Rev, given 平 | Rule effect | Imitation effect | Ends on 平 overall |")
    report.append("|---|---|---|---|---|---|---|---|")
    recall_ids = sorted(i for i, it in items.items() if it.get("set") == "recall")
    data["tone"] = {}
    writers = [("Classical originals", reference_cells(items, recall_ids))] + [(NAMES[m], tone_cells(rows, m)) for m in MODELS]
    for name, cells in writers:
        e = boot_effects(cells)
        p = e["p"]
        data["tone"][name] = {"rule": e["rule"], "imit": e["imit"], "ping": e["ping"], "cells": {f"{d}|{g}": p[(d, g)] for d, g in p}, "n": {f"{d}|{g}": e["n"][(d, g)] for d, g in e["n"]}}
        report.append(f"| {name} | {pct(p[('forward', ZE)])} | {pct(p[('forward', PING)])} | {pct(p[('reverse', ZE)])} | {pct(p[('reverse', PING)])} | "
                      f"{ci_str(*e['rule'])} | {ci_str(*e['imit'])} | {pct(e['ping'][0])} |")
    report.append("")

    report.append("### The same, with classical tones (平水韵)\n")
    report.append("Characters listed under both a 平 and a 仄 rhyme, or missing from the table, are left out (on either line).\n")
    report.append("| Writer | Fwd, given 仄 | Fwd, given 平 | Rev, given 仄 | Rev, given 平 | Rule effect | Imitation effect | Ends on 平 overall |")
    report.append("|---|---|---|---|---|---|---|---|")
    data["tone_classical"] = {}
    writers = [("Classical originals", reference_cells(items, recall_ids, "classical"))] + [(NAMES[m], tone_cells(rows, m, system="classical")) for m in MODELS]
    for name, cells in writers:
        e = boot_effects(cells)
        p = e["p"]
        data["tone_classical"][name] = {"rule": e["rule"], "imit": e["imit"], "ping": e["ping"], "n": {f"{d}|{g}": e["n"][(d, g)] for d, g in e["n"]}}
        report.append(f"| {name} | {pct(p[('forward', ZE)])} | {pct(p[('forward', PING)])} | {pct(p[('reverse', ZE)])} | {pct(p[('reverse', PING)])} | "
                      f"{ci_str(*e['rule'])} | {ci_str(*e['imit'])} | {pct(e['ping'][0])} |")
    report.append("")

    # the originals, novel by novel, in classical tones
    report.append("### The originals novel by novel (classical tones)\n")
    report.append("*Assignment*: the first line ends on 仄 and the second on 平 (the rule tested in phase 1). "
                  "*Opposition*: the two lines end on different tones, whichever way round. Couplets with an ambiguous or unlisted final character are left out.\n")
    report.append("| Work | Couplets | Assignment kept | Opposition kept | First line ends 仄 |")
    report.append("|---|---|---|---|---|")
    data["by_work"] = {}
    works = defaultdict(list)
    for it in items.values():
        a, b = final_class(it["first"], "classical"), final_class(it["second"], "classical")
        if a and b:
            works[it.get("work", it.get("set"))].append((a, b))
    for w, pairs in sorted(works.items(), key=lambda kv: -len(kv[1])):
        assign = np.mean([a == ZE and b == PING for a, b in pairs])
        opp = np.mean([a != b for a, b in pairs])
        first_ze = np.mean([a == ZE for a, _ in pairs])
        data["by_work"][w] = {"n": len(pairs), "assignment": float(assign), "opposition": float(opp), "first_ze": float(first_ze)}
        report.append(f"| {w} | {len(pairs)} | {pct(assign)} | {pct(opp)} | {pct(first_ze)} |")
    report.append("")

    # opposition: share of couplets whose two final characters differ in tone
    report.append("### Opposition: do the two lines end on different tones?\n")
    report.append("Share of answers whose final tone differs from the given line's (modern tones), against the originals.\n")
    report.append("| Writer | Recall set | Unseen set |")
    report.append("|---|---|---|")
    data["opposition"] = {}
    for name, m in [("Originals", None)] + [(NAMES[m], m) for m in MODELS]:
        res = {}
        for set_name in ("recall", "unseen"):
            vals = []
            if m is None:
                for it in items.values():
                    if it.get("set") != set_name:
                        continue
                    a, b = final_class(it["first"]), final_class(it["second"])
                    if a and b:
                        vals.append(a != b)
            else:
                for r in rows:
                    if r["model"] != m or r["set"] != set_name or r.get("end_modern_class") not in (PING, ZE):
                        continue
                    g = final_class(r["given"])
                    if g:
                        vals.append(g != r["end_modern_class"])
            res[set_name] = float(np.mean(vals)) if vals else None
        data["opposition"][name] = res
        report.append(f"| {name} | {pct(res['recall'])} | {pct(res['unseen'])} |")
    report.append("")

    # the same effects on the unseen set, where only the rule effect is identifiable
    report.append("### The same rule effect on the unseen set\n")
    report.append("Here the given line's tone is fixed by direction (given 仄 when writing second lines, 平 when writing first lines), "
                  "so the difference below mixes the rule effect with the imitation effect: rule − imitation, if both act.\n")
    report.append("| Writer | Ends 平, writing second lines | Ends 平, writing first lines | Difference |")
    report.append("|---|---|---|---|")
    for m in MODELS:
        f = [r["end_modern_class"] == PING for r in rows if r["model"] == m and r["set"] == "unseen" and r["direction"] == "forward" and r.get("end_modern_class") in (PING, ZE)]
        b = [r["end_modern_class"] == PING for r in rows if r["model"] == m and r["set"] == "unseen" and r["direction"] == "reverse" and r.get("end_modern_class") in (PING, ZE)]
        report.append(f"| {NAMES[m]} | {pct(np.mean(f))} | {pct(np.mean(b))} | {100 * (np.mean(f) - np.mean(b)):+.0f} |")
    report.append("")


# ------------------------------------------------------------------ 2. length and canvas

def section_length(rows, report, data):
    report.append("## 2. Length against the canvas\n")
    report.append("Signed error = 汉字 in the answer minus the required number, over non-empty answers.\n")
    report.append("| Model | Exact | Too short | Too long | Mean signed error | Mean absolute error |")
    report.append("|---|---|---|---|---|---|")
    data["length"] = {}
    for m in MODELS:
        errs = []
        for r in rows:
            if r["model"] != m or r.get("empty"):
                continue
            want = len(hanzi_only(r["given"]))  # the prompt asks for as many 汉字 as the given line has
            got = len(hanzi_only(r["answer"] or ""))
            if got == 0:
                continue
            errs.append(got - want)
        e = np.array(errs)
        data["length"][m] = {"exact": float(np.mean(e == 0)), "short": float(np.mean(e < 0)), "long": float(np.mean(e > 0)), "mean": float(e.mean()), "mae": float(np.abs(e).mean())}
        report.append(f"| {NAMES[m]} | {pct(np.mean(e == 0))} | {pct(np.mean(e < 0))} | {pct(np.mean(e > 0))} | {e.mean():+.2f} | {np.abs(e).mean():.2f} |")
    report.append("")

    # canvas slack on the dev set (calibration runs): does output length follow the canvas?
    report.append("### Does the output grow with the canvas? (dev set, calibration runs)\n")
    report.append("The canvas is the required number of characters plus a slack of 4, 6 or 8 tokens. "
                  "Content tokens = tokens before the first end-of-text token; 汉字 counted in the raw reply.\n")
    report.append("| Model | Slack | Samples | Content tokens / canvas | 汉字 minus required | Canvas filled to the end |")
    report.append("|---|---|---|---|---|---|")
    data["slack"] = {}
    for base, files in (("llada", [("4", "llada_T0.7.jsonl"), ("6", "llada_T0.7_slack6.jsonl"), ("8", "llada_T0.7_slack8.jsonl")]),
                        ("dream", [("4", "dream_T0.5.jsonl"), ("6", "dream_T0.5_slack6.jsonl"), ("8", "dream_T0.5_slack8.jsonl")])):
        for slack, fname in files:
            p = ROOT / "runs" / "calib" / fname
            if not p.exists():
                continue
            fill, over, full, n = [], [], 0, 0
            for line in open(p, encoding="utf-8"):
                rec = json.loads(line)
                if rec.get("type") == "header":
                    continue
                canvas = rec["settings"].get("canvas_tokens")
                for s in rec["samples"]:
                    ids = s.get("token_ids") or []
                    if not canvas or not ids:
                        continue
                    L = next((k for k, t in enumerate(ids) if t in END_TOKENS[base]), len(ids))
                    fill.append(L / canvas)
                    over.append(len(hanzi_only(s["raw"])) - rec["n_chars"])
                    full += L == len(ids)
                    n += 1
            data["slack"][f"{base}|{slack}"] = {"fill": float(np.mean(fill)), "over": float(np.mean(over)), "full": full / max(1, n), "n": n}
            report.append(f"| {NAMES[base]} | {slack} | {n} | {np.mean(fill):.2f} | {np.mean(over):+.2f} | {pct(full / max(1, n))} |")
    report.append("")


# ------------------------------------------------------------------ 3. order

def order_stats(model_label, base):
    ends = END_TOKENS[base]
    st = defaultdict(list)
    per_sample = {}
    for f in sorted(glob.glob(str(ROOT / "runs" / "full" / f"{model_label}_*.jsonl"))):
        hdr = None
        for line in open(f, encoding="utf-8"):
            rec = json.loads(line)
            if rec.get("type") == "header":
                hdr = rec
                continue
            if hdr["label"] != model_label:
                continue
            for k, s in enumerate(rec["samples"]):
                ids, steps = s.get("token_ids"), s.get("commit_step")
                if not ids or not steps or len(ids) != len(steps):
                    continue
                L = next((j for j, t in enumerate(ids) if t in ends), len(ids))
                if L < 2 or L == len(ids):
                    continue
                content = steps[:L]
                tail = steps[L:]
                order = np.argsort(np.argsort(content, kind="stable"), kind="stable")  # rank of each content position
                last_rank = order[-1] / (L - 1)  # 0 = filled first among content, 1 = filled last
                first_rank = order[0] / (L - 1)
                rho = float(np.corrcoef(np.arange(L), content)[0, 1]) if np.std(content) > 0 else np.nan
                tail_first = max(tail) < min(content)
                st["tail_first"].append(tail_first)
                st["last_rank"].append(last_rank)
                st["first_rank"].append(first_rank)
                st["rho"].append(rho)
                # where in the canvas was the very first commit?
                first_pos = int(np.argmin(steps))
                st["first_commit_in_tail"].append(first_pos >= L)
                per_sample[(model_label, rec["item_id"], rec["direction"], rec["condition"], k)] = {"last_rank": last_rank, "tail_first": tail_first}
    return st, per_sample


def section_order(rows, report, data):
    report.append("## 3. In which order the diffusion samplers fill the canvas\n")
    report.append("From `commit_step` (the denoising step at which each position was fixed). Samples whose answer is at least 2 tokens and ends inside the canvas. "
                  "*End padding first*: every token after the line (end-of-text and padding) was fixed before any token of the line, so the sampler settled the length before writing. "
                  "*Left-to-right* is the correlation between position and step within the line (+1 = strictly left to right, −1 = right to left). "
                  "*Last character's rank*: 0 if the line's last token was the first of the line to be fixed, 1 if it was the last.\n")
    report.append("| Model | Samples | First fixed token is padding | End padding first | Left-to-right (mean ρ) | Last token's rank | First token's rank |")
    report.append("|---|---|---|---|---|---|---|")
    data["order"] = {}
    linked = {}
    for m in ("llada", "llada-default", "dream", "dream-default"):
        base = "llada" if m.startswith("llada") else "dream"
        st, per = order_stats(m, base)
        linked.update(per)
        n = len(st["tail_first"])
        if not n:
            continue
        d = {k: float(np.nanmean(np.array(v, dtype=float))) for k, v in st.items()}
        d["n"] = n
        data["order"][m] = d
        report.append(f"| {NAMES[m]} | {n} | {pct(d['first_commit_in_tail'])} | {pct(d['tail_first'])} | {d['rho']:+.2f} | {d['last_rank']:.2f} | {d['first_rank']:.2f} |")
    report.append("")

    # link the last token's rank to the end-tone outcome
    report.append("### Is the end tone kept more often when the last character is fixed early?\n")
    report.append("End-tone pass rate (modern) by when the line's last token was fixed among the line's tokens. Descriptive only: "
                  "early and late samples differ in other ways too.\n")
    report.append("| Model | Direction | Fixed in the first third | Middle third | Last third |")
    report.append("|---|---|---|---|---|")
    data["order_link"] = {}
    for m in ("llada", "dream"):
        for direction in ("forward", "reverse"):
            buckets = defaultdict(list)
            for r in rows:
                if r["model"] != m or r["direction"] != direction:
                    continue
                key = (m, r["item_id"], r["direction"], r["condition"], r["sample"])
                info = linked.get(key)
                if info is None or r.get("rule_tone_modern") is None:
                    continue
                b = 0 if info["last_rank"] < 1 / 3 else (1 if info["last_rank"] < 2 / 3 else 2)
                buckets[b].append(1.0 if r["rule_tone_modern"] else 0.0)
            cells = [f"{pct(np.mean(buckets[b]))} (n={len(buckets[b])})" if buckets[b] else "–" for b in range(3)]
            data["order_link"][f"{m}|{direction}"] = {str(b): (float(np.mean(buckets[b])) if buckets[b] else None, len(buckets[b])) for b in range(3)}
            report.append(f"| {NAMES[m]} | {direction} | {cells[0]} | {cells[1]} | {cells[2]} |")
    report.append("")


# ------------------------------------------------------------------ 4. copying

def section_copy(rows, items, report, data):
    report.append("## 4. Characters taken from the summary and from the given line\n")
    report.append("With-summary condition, both sets. *From summary*: share of the answer's 汉字 that also occur in the summary. "
                  "*From given line*: share that also occur in the given line. The originals are scored the same way against their own summary. "
                  "The last column is the correlation, across samples, between the summary share and the relevance score.\n")
    report.append("| Writer | Answers | From summary | From given line | Relevance | Corr(summary share, relevance) |")
    report.append("|---|---|---|---|---|---|")
    data["copy"] = {}
    for m in MODELS:
        fs, fg, rel = [], [], []
        for r in rows:
            if r["model"] != m or r["condition"] != "summary" or r.get("empty"):
                continue
            ans = hanzi_only(r["answer"] or "")
            summ = items[r["item_id"]].get("summary") or ""
            if not ans or not summ:
                continue
            fs.append(np.mean([c in summ for c in ans]))
            fg.append(np.mean([c in r["given"] for c in ans]))
            rel.append(r["relevance"] if r.get("relevance") is not None else np.nan)
        fs, fg, rel = np.array(fs), np.array(fg), np.array(rel, dtype=float)
        ok = ~np.isnan(rel)
        corr = float(np.corrcoef(fs[ok], rel[ok])[0, 1]) if ok.sum() > 2 else float("nan")
        data["copy"][m] = {"from_summary": float(fs.mean()), "from_given": float(fg.mean()), "relevance": float(np.nanmean(rel)), "corr": corr, "n": int(len(fs))}
        report.append(f"| {NAMES[m]} | {len(fs)} | {pct(fs.mean())} | {pct(fg.mean())} | {np.nanmean(rel):.2f} | {corr:+.2f} |")
    fs, fg = [], []
    for it in items.values():
        if it.get("set") not in ("unseen", "recall") or not it.get("summary"):
            continue
        for written, given in ((it["second"], it["first"]), (it["first"], it["second"])):
            ans = hanzi_only(written)
            fs.append(np.mean([c in it["summary"] for c in ans]))
            fg.append(np.mean([c in given for c in ans]))
    data["copy"]["originals"] = {"from_summary": float(np.mean(fs)), "from_given": float(np.mean(fg)), "n": len(fs)}
    report.append(f"| Originals | {len(fs)} | {pct(np.mean(fs))} | {pct(np.mean(fg))} | – | – |")
    report.append("")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = load_rows()
    items = load_items()
    report = ["# Exploratory analyses of existing outputs\n",
              "Post hoc; not part of the frozen plan. Generated by `explore/analyse_existing.py` from `results/full_scored.jsonl`, "
              "`runs/full/`, `runs/calib/` and the item files. See `PROTOCOL.md` entry 36.\n"]
    data = {}
    section_tone(rows, items, report, data)
    section_length(rows, report, data)
    section_order(rows, report, data)
    section_copy(rows, items, report, data)
    (OUT / "existing_data.md").write_text("\n".join(report) + "\n", encoding="utf-8", newline="\n")
    (OUT / "existing_data.json").write_text(json.dumps(data, ensure_ascii=False, indent=1, default=float) + "\n", encoding="utf-8", newline="\n")
    print("\n".join(report))
    return 0


if __name__ == "__main__":
    sys.exit(main())
