# Frozen plan for the phase 1 full run

Copied verbatim from PROTOCOL.md entries 28 and 29 at the freeze, 2026-11-21 18:26. The author approved all five arms ("Yes, all five arms", 2026-11-21).

### 28. Analysis plan for the full run (planned; to be frozen with the settings)

Written before any test item has been generated on.

**What is run.** Each model on four parts: unseen with summary (43 items), unseen bare (43), recall with summary (48), recall bare (580). Both directions, 8 samples each. The bare unseen part is an addition to the brief: it costs minutes and makes a clean two-by-two (set × condition), so that "unseen against recall" is never also "summary against no summary".

**Questions, in order of weight.**

1. *New couplets.* On the unseen set with summaries: does either diffusion model beat Qwen on all-rules pass rate, on parallelism, on relevance? (Paired by item.)
2. *Memorized couplets.* On the recall set, bare: exact-match rate per model and direction. This is the nearest thing to the LLaDA paper's poem test: is the reverse direction (recalling the earlier line) where diffusion does relatively better?
3. *Does any advantage survive?* The diffusion-minus-Qwen difference on the unseen set minus the same difference on the recall set, in matched conditions.
4. *Unseen minus recall*, per model and direction, for every metric (the gap the brief asks for).

**Reading rule.** A difference is called a difference only if its 95% bootstrap interval over items excludes zero. With 43 unseen items the intervals are wide; "no detectable difference" will be said as that, not as "the same". There are many comparisons and no correction for multiplicity, so a lone starred cell among many is weak evidence; patterns that hold across both directions and both conditions count for more.

**Not claimed.** Nothing about literary quality from the automatic scores; that is what the 51 graded lines are for. Nothing about diffusion models in general from two 7–8B models.

**Human grading (51 lines).** From the unseen set, with-summary condition. 17 cells (chapter, direction) drawn with seed 20261119: 9 forward and 8 reverse, no chapter twice. For each cell one line per model, drawn at random from that model's samples that pass all rules; if none passes, from all its non-empty samples, and the key records which. Shuffled; the sheet shows the summary and the couplet with the generated line marked, never the model. The grade therefore answers: "when a model does produce a line that keeps the rules, how good is it?" How often it produces one is the automatic pass rate. `scripts/make_grading_sheet.py` builds the sheet; `scripts/unblind.py` refuses to read the key until `grades.json` exists.

### 29. Proposed freeze (awaiting the author at checkpoint 2)

Nothing has been generated on the unseen or recall sets. On the author's go-ahead, `scripts/freeze.py --write` records the SHA-256 of every file below and the full run starts; after that, a change to any of them needs a dated entry here.

- **Sets**: unseen v1 (43), recall (580, of which 48 with summaries), dev and its splits; the 平水韵 table.
- **Settings** (`config/phase1.json`): prompt layout v2; canvas of required characters + 6 tokens; one denoising step per canvas token; 8 samples per item in two batches of 4; base seed 20261119; bf16. Arms: Qwen at vendor defaults; Dream 0.5 and LLaDA 0.7 (primary, matched diversity); Dream 0.2 and LLaDA 0 (vendor defaults; a robustness arm the author may drop).
- **Code**: prompt, extraction, checker, tones, parallelism (LTP base2), scoring, statistics, generation, loading, registry, run and report scripts, grading-sheet builder; `uv.lock`.
- **Plan**: entry 28.

Estimated GPU time for the full run, from the pilot's timings: Qwen 25 minutes, Dream 1.7 hours per arm, LLaDA 2 hours for the primary arm and 25 minutes for the deterministic one. Three primary arms: about 4.2 hours. All five: about 6.3 hours.
