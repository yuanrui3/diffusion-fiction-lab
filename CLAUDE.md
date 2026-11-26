# diffusion-fiction-lab: working rules

Read this file and `PROTOCOL.md` before doing anything. The author's background report is one folder up, at `../docs/background-research.md`.

## The question (phase 1 of 3)

Do diffusion language models write Chinese 回目 couplets better than an autoregressive model of similar size, and does any advantage survive when the couplet is new rather than memorized?

- Phase 1 (this one): 回目 couplets, unseen set (《火候》) against a recall set (public-domain 章回 novels).
- Phase 2 (later, do not start): foreshadowing, filling in a setup whose payoff comes later.
- Phase 3 (later, do not start): same weights, left-to-right against diffusion order, for the author's philosophizing benchmark.

Keep model loading and generation in `src/dfl/` reusable for phases 2 and 3: fixed anchors inside the canvas and a choice of unmasking order are part of the generation interface for that reason.

## Hard constraints

- No paid APIs. Only open-weight models run locally on this machine, and free local tools.
- The only human judge is the author, for about 50 blinded items at the end. Everything else is scored automatically.
- One GPU, one model loaded at a time. Run `nvidia-smi` and check free disk space before any download.
- Use only the public 洁本 text of 《火候》 (GitHub `slowfire-novel/Slow-fire-novel`, folder `火候/`). Nothing private goes into this repo: no full-edition text, no email addresses, no personal details. Chapter text itself is not copied here, only the 回目, the summaries and the source commit.
- Verify Hugging Face IDs and licences before downloading a model, and pin the commit.

## How to work

- `PROTOCOL.md` is the lab notebook: every decision, what was tried, what was kept and why, in order, dated. Append; do not rewrite history. If a past decision turns out wrong, add a new entry that says so.
- Fix the metrics and the test sets before the full run. Do not change them after seeing results. Tuning happens only on the dev set (`data/dev/`), never on the unseen or recall sets.
- Record for every run: model ID and commit, numeric precision, seed, sampling settings, number of denoising steps, canvas size, prompt version, code commit.
- Stop and report briefly to the author at three points: (1) after setup and a smoke test of all three models on 3 items, (2) after a 10-item pilot with the checker, (3) after the full run.
- The author does not micromanage: make reasonable calls and record them in `PROTOCOL.md`.
- Talk to the author in English. Any Chinese written for this project (summaries, prompts, sheet labels) must read as natural native Chinese, never translated-sounding.
- The chapter summaries in `data/unseen/` were written by Claude and are frozen. Do not rewrite them; if one is wrong, record the correction in `PROTOCOL.md` and bump the set version.
- Commit and push only when the author asks. The remote is the public repo `yuanrui3/diffusion-fiction-lab`, so anything committed is published: never commit the grading key before grading is done, chapter text, or local paths. Commits use the repo-local GitHub no-reply identity.

## Models

| Role | Hugging Face ID | Commit | Licence |
|---|---|---|---|
| Diffusion, full canvas | `GSAI-ML/LLaDA-8B-Instruct` | `08b83a6feb34df1a6011b80c3c00c7563e963b07` | MIT |
| Diffusion, full canvas, from Qwen2.5-7B | `Dream-org/Dream-v0-Instruct-7B` | `05334cb9faaf763692dcf9d8737c642be2b2a6ae` | Apache-2.0 |
| Autoregressive control | `Qwen/Qwen2.5-7B-Instruct` | `a09a35458c702b33eeacc393d103063234e8bc28` | Apache-2.0 |
| Embeddings for relevance | `BAAI/bge-m3` | `5617a9f61b028005a4858fdac845db406aefb181` | MIT |
| Optional later | `inclusionAI/LLaDA2.0-mini` (32.5 GB, 8-bit only) | `dad945cac317da394b390f82c7b40691d8a881ed` | Apache-2.0 |

LLaDA and Dream load with `trust_remote_code=True`; their custom code was read at these commits before first use. Always pass `revision=` so a later upstream change cannot alter a run.

## This machine

- GPU: NVIDIA RTX 4090 Laptop, 16 GB VRAM (not the 24 GB the author expected). See `PROTOCOL.md` for how precision is handled.
- 64 GB RAM, i9-13980HX. Disk: C: about 2.2 TB free, D: about 816 GB free (checked 2026-11-19).
- Windows 11, Git Bash and PowerShell. Chinese characters in `curl` arguments get mangled by the console code page: percent-encode URLs and use ASCII file names for downloads.
- Doubled backslashes typed into a shell command arrive collapsed. Put any script that needs them in a file and run the file.
- Python: the project environment is `.venv` (Python 3.11, built by `uv` from `pyproject.toml` and `uv.lock`). `uv` itself is at `../_tools/uv/uv.exe`, not on PATH. There is no system Python.
- Model weights are plain folders in `../_models/<key>/` (the Hugging Face cache cannot make symlinks here). `scripts/download_models.py` fetches them at the pinned commits.
- All three models run in bf16 with the input embedding table on the CPU. Keep 4 samples per batch: 8 spills past the 16 GB card.

## Commands

Run from the repo root, in Git Bash:

```bash
export PYTHONUTF8=1                                   # the console cannot print Chinese without it
.venv/Scripts/python.exe -m pytest -q                 # unit tests
.venv/Scripts/python.exe scripts/build_unseen.py      # verify the frozen unseen set against the source
.venv/Scripts/python.exe scripts/run_generation.py --model qwen --items data/dev/dev.jsonl --ids jhy-003 --condition bare --out runs/smoke/qwen.jsonl
.venv/Scripts/python.exe scripts/score.py "runs/smoke/*.jsonl" --show 4 --relevance
```

One model per process. Never start a second generation while one is running, and do not score with `--relevance` or build reports (both load small models on the GPU) while a generation is running.

## Pipeline, in order

| Step | Script | Writes |
|---|---|---|
| Download weights | `scripts/download_models.py` | `../_models/` |
| Classical tone table | `scripts/build_pingshui.py` | `data/lexicon/pingshui.json` |
| Recall set and dev pool | `scripts/build_recall.py` | `data/recall/recall.jsonl`, `data/dev/dev_pool.jsonl` |
| Recall chapters to summarise | `scripts/fetch_recall_chapters.py` | `data/recall/summary_sample.json`, `data/raw/recall_chapters/` |
| Merge recall summaries | `scripts/merge_recall_summaries.py --write` | `data/recall/recall_summaries.json`, fills `recall.jsonl` |
| Dev set and its splits | `scripts/build_dev.py` | `data/dev/dev.jsonl`, `data/dev/splits.json` |
| Verify the unseen set | `scripts/build_unseen.py` | nothing (check only) |
| Choose the tagger | `scripts/compare_taggers.py` | nothing (prints the comparison) |
| Calibrate temperature | `scripts/calibrate.py run`, then `report` | `runs/calib/` |
| Run a plan | `scripts/run_plan.py pilot` or `full` | `runs/<plan>/` |
| Score | `scripts/score.py "runs/<plan>/*.jsonl" --relevance --out results/<plan>_scored.jsonl` | scored rows |
| Report tables | `scripts/report.py results/<plan>_scored.jsonl --out results/<plan>_report.md` | Markdown tables |
| Grading sheet | `scripts/make_grading_sheet.py results/full_scored.jsonl` | `results/grading/` |
| Unblind the author's grades (needs `grades.json`) | `scripts/unblind.py` | `results/grading/graded.jsonl`, `grades_report.md` |

Phase 1 finished on 2026-11-25: the answer is in `results/results.md`, the last entry in `PROTOCOL.md` is 34.

The frozen settings for phase 1 live in `config/phase1.json`; `run_plan.py` reads nothing else. The parallelism score uses the LTP tagger pinned in `src/dfl/registry.py` (free for research and personal use, not for commercial use).

`results/grading/KEY_DO_NOT_OPEN.json` says which model wrote which graded line. Do not open it, print it or quote it before the author's `grades.json` exists.

## Layout

- `src/dfl/`: the reusable package (checker, tones, prompts, answer extraction, model loading, generation).
- `scripts/`: one script per step (build the sets, run generation, score, make the grading sheet).
- `tests/`: unit tests for the checker and the extraction rules.
- `data/unseen/`, `data/recall/`, `data/dev/`: the test sets and the tuning set. `data/lexicon/`: tone tables.
- `runs/`: raw generations, one JSONL per model and set, with the run settings inside.
- `results/`: scored tables, the grading sheet, `results.md`.
