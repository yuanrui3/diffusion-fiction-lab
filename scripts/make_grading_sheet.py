"""Build the blinded grading sheet for the author.

Usage:
    python scripts/make_grading_sheet.py results/full_scored.jsonl
    python scripts/make_grading_sheet.py results/pilot_scored.jsonl --set dev --out results/grading_test   # dry run

Selection (fixed in PROTOCOL.md before the full run):
  1. Cells are (chapter, direction) pairs of the chosen set in the with-summary
     condition. 17 cells are drawn with a fixed seed: 9 forward and 8 reverse,
     no chapter twice.
  2. For each cell, one line per model: drawn at random from that model's
     samples that pass all rules; if none passes, from all its non-empty
     samples (recorded in the key).
  3. The 51 entries are shuffled.

Writes, into the output folder:
  grading_sheet.html       the sheet: opens offline in a browser, saves as you go, exports grades.json
  KEY_DO_NOT_OPEN.json     which model wrote which entry (not to be opened before grading is done)
"""

from __future__ import annotations

import argparse
import html
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ITEM_FILES = {"unseen": "data/unseen/huohou.jsonl", "recall": "data/recall/recall.jsonl", "dev": "data/dev/dev.jsonl"}
SEED = 20261119
N_FORWARD, N_REVERSE = 9, 8
MODELS = ["qwen", "dream", "llada"]

PAGE = """<!doctype html>
<html lang="zh-Hans">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>回目 grading sheet</title>
<style>
  :root { --bg:#faf8f4; --card:#ffffff; --ink:#1f1d1a; --muted:#6b665e; --line:#e3ded4; --accent:#8a3b12; --mark:#fff3d6; }
  @media (prefers-color-scheme: dark) {
    :root { --bg:#181715; --card:#22201d; --ink:#ece8e1; --muted:#a39d92; --line:#3a3631; --accent:#e0a070; --mark:#4a3a18; }
  }
  * { box-sizing: border-box; }
  body { margin:0; background:var(--bg); color:var(--ink); font:16px/1.6 "Noto Serif SC","Source Han Serif SC","Songti SC",SimSun,serif; }
  header, main { max-width: 760px; margin: 0 auto; padding: 16px; }
  h1 { font: 600 20px/1.3 system-ui, sans-serif; margin: 8px 0; }
  p.note, .ui { font: 14px/1.5 system-ui, sans-serif; color: var(--muted); }
  .bar { position: sticky; top: 0; background: var(--bg); border-bottom: 1px solid var(--line); padding: 8px 16px; z-index: 2; }
  .bar div { max-width: 760px; margin: 0 auto; display: flex; gap: 12px; align-items: center; justify-content: space-between; }
  button { font: 14px system-ui, sans-serif; padding: 6px 12px; border: 1px solid var(--accent); background: transparent; color: var(--accent); border-radius: 6px; cursor: pointer; }
  .card { background: var(--card); border: 1px solid var(--line); border-radius: 10px; padding: 16px; margin: 16px 0; }
  .card.done { border-color: var(--accent); }
  .num { font: 600 13px system-ui, sans-serif; color: var(--muted); }
  .summary { color: var(--muted); font-size: 15px; margin: 6px 0 12px; }
  .couplet { font-size: 22px; letter-spacing: 0.12em; margin: 4px 0; }
  .couplet .tag { font: 12px system-ui, sans-serif; letter-spacing: 0; color: var(--muted); margin-right: 8px; }
  .couplet.judged span.line { background: var(--mark); padding: 0 4px; border-radius: 4px; }
  .scores { display: flex; gap: 6px; flex-wrap: wrap; margin: 12px 0 8px; }
  .scores label { font: 14px system-ui, sans-serif; border: 1px solid var(--line); border-radius: 6px; padding: 4px 12px; cursor: pointer; }
  .scores input { display: none; }
  .scores input:checked + span { font-weight: 700; color: var(--accent); }
  .scores label:has(input:checked) { border-color: var(--accent); }
  textarea { width: 100%; min-height: 44px; font: 15px/1.5 inherit; padding: 6px 8px; border: 1px solid var(--line); border-radius: 6px; background: transparent; color: inherit; }
</style>
</head>
<body>
<div class="bar"><div>
  <span class="ui" id="progress"></span>
  <button id="export">Export grades.json</button>
</div></div>
<header>
  <h1>回目 grading sheet</h1>
  <p class="note">__COUNT__ lines to grade. Each card shows the chapter summary and a couplet. One line is the original and was given to the model; the highlighted line was written by a model. Score the highlighted line for literary quality, 1 to 5, as a line of this couplet. Which model wrote it is hidden, and the order is shuffled.</p>
  <p class="note">Suggested anchors: 1 broken or meaningless · 2 reads, but flat or off the chapter · 3 acceptable · 4 good · 5 could stand as the chapter's 回目. Comments are optional. Your marks are saved in this browser as you go; press Export when done and put <code>grades.json</code> in <code>results/grading/</code>.</p>
</header>
<main id="cards"></main>
<script>
const ENTRIES = __ENTRIES__;
const KEY = "huimu-grades-__SHEET_ID__";
let saved = {};
try { saved = JSON.parse(localStorage.getItem(KEY) || "{}"); } catch (e) { saved = {}; }
function store() { try { localStorage.setItem(KEY, JSON.stringify(saved)); } catch (e) {} update(); }
function update() {
  const done = ENTRIES.filter(e => saved[e.entry] && saved[e.entry].score).length;
  document.getElementById("progress").textContent = done + " of " + ENTRIES.length + " graded";
  ENTRIES.forEach(e => document.getElementById("card-" + e.entry).classList.toggle("done", !!(saved[e.entry] && saved[e.entry].score)));
}
const main = document.getElementById("cards");
ENTRIES.forEach((e, i) => {
  const card = document.createElement("div");
  card.className = "card"; card.id = "card-" + e.entry;
  const row = (tag, text, judged) => '<div class="couplet' + (judged ? ' judged' : '') + '"><span class="tag">' + tag + (judged ? ' · to grade' : ' · given') + '</span><span class="line">' + text + '</span></div>';
  let scores = "";
  for (let s = 1; s <= 5; s++) scores += '<label><input type="radio" name="s-' + e.entry + '" value="' + s + '"><span>' + s + '</span></label>';
  card.innerHTML = '<div class="num">' + (i + 1) + ' / ' + ENTRIES.length + '</div>' +
    '<div class="summary">' + e.summary + '</div>' +
    row("上句", e.first, e.judged === "first") + row("下句", e.second, e.judged === "second") +
    '<div class="scores">' + scores + '</div>' +
    '<textarea placeholder="comment (optional)"></textarea>';
  main.appendChild(card);
  const prev = saved[e.entry] || {};
  card.querySelectorAll("input").forEach(inp => {
    if (String(prev.score) === inp.value) inp.checked = true;
    inp.addEventListener("change", () => { saved[e.entry] = Object.assign(saved[e.entry] || {}, {score: Number(inp.value)}); store(); });
  });
  const ta = card.querySelector("textarea");
  ta.value = prev.comment || "";
  ta.addEventListener("input", () => { saved[e.entry] = Object.assign(saved[e.entry] || {}, {comment: ta.value}); store(); });
});
update();
document.getElementById("export").addEventListener("click", () => {
  const out = {sheet: "__SHEET_ID__", exported: new Date().toISOString(),
               grades: ENTRIES.map(e => ({entry: e.entry, score: (saved[e.entry] || {}).score || null, comment: (saved[e.entry] || {}).comment || ""}))};
  const blob = new Blob([JSON.stringify(out, null, 1)], {type: "application/json"});
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob); a.download = "grades.json"; a.click();
});
</script>
</body>
</html>
"""


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("scored", type=Path)
    ap.add_argument("--set", default="unseen")
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "grading")
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    rows = [json.loads(line) for line in args.scored.read_text(encoding="utf-8").splitlines() if line.strip()]
    rows = [r for r in rows if r["set"] == args.set and r["condition"] == "summary"]
    items = {}
    for line in (ROOT / ITEM_FILES[args.set]).read_text(encoding="utf-8").splitlines():
        if line.strip():
            it = json.loads(line)
            items[it["id"]] = it

    by_cell: dict[tuple, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        by_cell[(r["item_id"], r["direction"])][r["model"]].append(r)
    complete = sorted(cell for cell, per_model in by_cell.items() if all(per_model.get(m) for m in MODELS))

    rng = random.Random(args.seed)
    item_ids = sorted({cell[0] for cell in complete})
    rng.shuffle(item_ids)
    want = {"forward": N_FORWARD, "reverse": N_REVERSE}
    if len(item_ids) < N_FORWARD + N_REVERSE:          # a small set (dry run): use what there is
        want = {"forward": (len(item_ids) + 1) // 2, "reverse": len(item_ids) // 2}
    cells = []
    for item_id in item_ids:
        for direction in ("forward", "reverse"):
            if want[direction] > 0 and (item_id, direction) in by_cell and (item_id, direction) in complete:
                cells.append((item_id, direction))
                want[direction] -= 1
                break
    if any(want.values()):
        print(f"could not fill the quota: {want} left")
        return 1

    entries, key = [], {}
    for item_id, direction in cells:
        it = items[item_id]
        for model in MODELS:
            samples = sorted(by_cell[(item_id, direction)][model], key=lambda r: r["sample"])
            passing = [r for r in samples if r["all_rules_modern"]]
            pool = passing or [r for r in samples if r["answer"]] or samples
            pick = rng.choice(pool)
            entry = {"item_id": item_id, "direction": direction, "model": model, "sample": pick["sample"],
                     "answer": pick["answer"], "passed_all_rules": bool(pick["all_rules_modern"]),
                     "n_passing_of_8": len(passing)}
            entries.append(entry)
    rng.shuffle(entries)

    sheet_id = f"{args.set}-{args.seed}"
    public = []
    for k, e in enumerate(entries, 1):
        code = f"e{k:02d}"
        it = items[e["item_id"]]
        judged = "second" if e["direction"] == "forward" else "first"
        first = e["answer"] if judged == "first" else it["first"]
        second = e["answer"] if judged == "second" else it["second"]
        public.append({"entry": code, "summary": html.escape(it["summary"]), "first": html.escape(first),
                       "second": html.escape(second), "judged": judged})
        key[code] = e

    args.out.mkdir(parents=True, exist_ok=True)
    page = (PAGE.replace("__ENTRIES__", json.dumps(public, ensure_ascii=False))
                .replace("__SHEET_ID__", sheet_id).replace("__COUNT__", str(len(public))))
    (args.out / "grading_sheet.html").write_text(page, encoding="utf-8", newline="\n")
    (args.out / "KEY_DO_NOT_OPEN.json").write_text(
        json.dumps({"sheet": sheet_id, "seed": args.seed, "scored_file": str(args.scored.as_posix()), "key": key},
                   ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n")
    n_fallback = sum(not e["passed_all_rules"] for e in entries)
    print(f"sheet: {len(public)} entries from {len(cells)} cells ({args.set}); "
          f"{n_fallback} entries had no rule-passing sample to draw from")
    print(f"wrote {args.out / 'grading_sheet.html'} and the key")
    return 0


if __name__ == "__main__":
    sys.exit(main())
