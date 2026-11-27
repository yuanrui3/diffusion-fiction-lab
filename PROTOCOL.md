# Protocol log

Lab notebook for phase 1 (回目 couplets). Entries are in order and are not rewritten; corrections are new entries. "Decided" means in force. "Planned" means proposed and still open until the freeze entry that will precede the full run.

---

## 2026-11-19

### 1. Starting point

Read `../docs/background-research.md`. What this phase takes from it:

- The LLaDA paper's poem test (previous line 45.6 for LLaDA-8B-Instruct, 38.0 for Qwen2.5-7B, 34.3 for GPT-4o; next line 51.8, 75.9, 82.7) measures recall of memorized lines, not composition under 平仄 and 对仗.
- No published work applies a diffusion language model to 对联 or 回目.
- PoetryDiffusion's metre scores came from classifier guidance, not from diffusion as such. So this phase keeps guidance out of the main comparison: no rule-aware decoding for any model.
- Dream-7B was initialized from Qwen2.5-7B, which makes Dream against Qwen2.5-7B-Instruct the closest pair by lineage.

### 2. Machine check

- `nvidia-smi`: one NVIDIA GeForce RTX 4090 Laptop GPU, **16,376 MiB** VRAM, driver 610.74, nothing else using it. The author expected 24 GB.
- 64 GB RAM, Intel i9-13980HX. Free disk: C: 2.2 TB, D: 816 GB. Three 7–8B models need about 47 GB plus 2.3 GB for bge-m3.
- No Python on the machine (only the Microsoft Store stub), no uv, no conda. `winget` and `git` are present; `gh` and `node` are not.
- Tried to download the `uv` package manager (official GitHub release) into a local tools folder. The session's permission system blocked it. **Open: needs the author's go-ahead before any Python can run.** Until then only data collection and writing code are possible; nothing below has been executed in Python yet.

Consequence of 16 GB (planned, to be confirmed in the smoke test): bf16 weights are 16.0 GB for LLaDA and 15.2 GB for Dream and Qwen, so none fits with room to spare. Plan, in order of preference:

1. bf16 for all three, with the input embedding matrix kept on the CPU (about 1 GB each; a lookup table, so the values are identical and the cost is negligible). Expected GPU use about 13–14 GB.
2. If that does not fit or spills into shared memory, 8-bit weights (bitsandbytes) for all three.

Whatever is chosen is applied to all three models alike. A mixed setup (one model quantized, another not) is ruled out, because the cited quantization study found Dream loses more than LLaDA.

### 3. Models verified (Hugging Face API, 2026-11-19)

| Model | Commit | Licence | Size | Custom code |
|---|---|---|---|---|
| `GSAI-ML/LLaDA-8B-Instruct` | `08b83a6f…963b07` | MIT | 16.04 GB | yes |
| `Dream-org/Dream-v0-Instruct-7B` | `05334cb9…b6ae` | Apache-2.0 | 15.24 GB | yes |
| `Qwen/Qwen2.5-7B-Instruct` | `a09a3545…bc28` | Apache-2.0 | 15.24 GB | no |
| `BAAI/bge-m3` | `5617a9f6…b181` | MIT | 2.27 GB (PyTorch weights only) | no |
| `inclusionAI/LLaDA2.0-mini` (optional) | `dad945ca…81ed` | Apache-2.0 | 32.52 GB | yes |

None is gated. Full commit hashes are in `CLAUDE.md`. The custom code of LLaDA (`modeling_llada.py`) and Dream (`modeling_dream.py`, `generation_utils.py`), and LLaDA's reference sampler (`generate.py` in `ML-GSAI/LLaDA`), were fetched as text at those commits and read before use.

Vendor default sampling, as found:

- Qwen2.5-7B-Instruct `generation_config.json`: sampling on, temperature 0.7, top_p 0.8, top_k 20, repetition_penalty 1.05.
- LLaDA reference sampler: temperature 0, low-confidence remasking, steps equal to canvas length in its examples.
- Dream `generation_config.json`: temperature 0, algorithm `origin`, 512 steps.

Both diffusion defaults are deterministic, so 8 samples per item would be 8 copies. Sampling temperature for the diffusion models therefore has to be chosen; see entry 8.

### 4. Unseen set: source and count

- Source: `https://github.com/slowfire-novel/Slow-fire-novel`, commit `060aab85af894ba0ccdef754634cd7e0365e9176` (2026-11-19), folder `火候/chapters/`. The repo's README states this folder is the public 洁本 edition.
- 45 files: chapters 1–44 and 外一章. The 回目 is the first non-empty line under the chapter heading, two lines separated by a full-width space.
- **43 couplets.** Chapter 24 (门廊) and 外一章 (野地) have no 回目 and are left out.
- Line lengths: 7 characters in 29 chapters, 8 in 12, 9 in one (ch. 27), 6 in one (ch. 44).
- The chapter text is not copied into this repo. A clone used for reading sits outside the repo at `../_sources/Slow-fire-novel`.

### 5. Chapter summaries (written by Claude)

Decided:

- One summary per chapter with a 回目, 43 in all, written once and then frozen.
- Method: six Claude subagents each read a run of consecutive chapters in full from the 洁本 files and wrote the summaries under the rules below. The main session checks every summary against the rules and reads for natural Chinese; it does not re-read all chapters. Raw agent output is kept in `data/unseen/summaries_raw/`.
- The summariser sees the 回目 (it sits under the chapter title). The rules require both events the 回目 points to be present in the summary, in different words. Reason: without the event, a model cannot write a line about it, and the task would test guessing; with the same wording, it would test copying.
- Rules given to the summarisers, verbatim:

  1. 两到三句，合计 60–110 个汉字。用平实的现代书面语：不用文言，不堆四字套话，读起来要像中国人自己写的内容提要，不能有翻译腔。
  2. 只交代这一章里发生了什么：谁，在哪儿，做了什么，结果怎样。不评论，不抒情，不点题，不引原文对白。
  3. 一章通常有两条线，回目上下两句各指一件事。两件事都要写到，但必须用你自己的话：除人名、地名之外，梗概里不得出现和回目任何一句相同的连续三个字（含）以上。
  4. 人名、地名照书里的写法。头一回出现的人，可以顺手带上身份，但不要为交代身份把句子撑长。
  5. 不提章名，不出现“回目”“本章”“这一章”之类的字眼。
  6. 书里人物说成都话，梗概一律用普通话，方言词不进梗概。
  7. 这是洁本，写到男女之事只需一笔带过，不要展开。

- Automatic check before freezing: no run of three or more characters shared between a summary and its 回目, other than a personal or place name. Exceptions are listed by hand in the set file.

Known limits, to carry into `results.md`: the summaries are by Claude, and the 回目 themselves were drafted with Claude's help, so the summaries may sit closer to the 回目 than an independent reader's would.

### 6. Recall set: source (planned, not yet built)

- 回目 from the contents pages of Chinese Wikisource (public-domain texts; site content CC BY-SA 4.0): 紅樓夢 (120 回), 水滸傳 (120回本), 三國演義 (毛宗崗本, 120 回), 西遊記 (100 回), 儒林外史 (56 回), 海上花列傳 (64 回). About 580 couplets. All six contents pages were fetched on 2026-11-19 and list the 回目 in one consistent line format.
- 水滸傳: the 120-chapter edition, because it contains the 100-chapter text plus 20 chapters and gives more items.
- Wikisource text is in traditional characters. It will be converted to simplified (OpenCC `t2s`), with a hand check of the characters known to convert ambiguously. The traditional original is kept beside it.
- Where Wikisource gives variant readings with the `{{另|A|B}}` template (西遊記), the first reading is used.
- Items whose two lines differ in length in the source stay in the set; the reference couplet simply fails the length rule, which is part of the ceiling.

Summaries for the recall set (planned): a seeded, stratified sample of chapters gets a Claude-written summary under the same seven rules, from the Wikisource chapter text, so that the with-summary condition matches the unseen set. The rest of the recall set is run with the 回目 alone. The size of the sample is set at the freeze.

### 7. Dev set (planned)

回目 from 《鏡花緣》 (Wikisource, 100 回), which is in neither test set. All tuning (canvas slack, diffusion temperature, prompt wording, extraction rules) and the smoke test happen here. The unseen and recall sets are not generated on until the freeze.

This changes one thing from the brief: the smoke test (3 items) and the pilot (10 items) run on dev items, not on test items, so that nothing is tuned on the data the result depends on.

### 8. Task design (planned, to be frozen before the full run)

**Item and directions.** Each item is one couplet. Forward: given the first line, write the second. Reverse: given the second, write the first. 8 samples per item, per direction, per model.

**Prompt.** One user message, no custom system prompt, each model's own chat template. The same text for all three models. It states every rule the checker applies: the length in characters, parallel structure, the tone of each line's last character, no reuse of the given line's characters, and (when there is a summary) fit to the summary. It asks for the line alone. Draft wording is in `src/dfl/prompts.py`.

**Tokens against characters (diffusion canvas).** Both tokenizers merge common character pairs into one token and may split a rare character into several. So a line of L characters takes at most about L tokens and often fewer, and no canvas size can force exactly L characters without restricting the vocabulary. Planned handling:

- Main arm: the reply canvas is L + 4 masked tokens. The slack leaves room for the end-of-turn token and for rare characters. The model decides where the line ends by writing end tokens, exactly as the autoregressive model does. Length is then a real test for all three models.
- The canvas size never depends on the reference answer's tokenization, which would leak its length in tokens.
- Possible second arm, to be offered to the author and not built unless wanted: restrict every model to single-character tokens and exactly L positions, which makes the length rule automatic and compares the models on the remaining rules only.

**Denoising steps.** Steps equal the canvas length, so about one token is committed per step, the highest-quality setting each sampler offers. Each model uses its authors' sampler: LLaDA low-confidence remasking; Dream `entropy` ordering with top_p 0.95 as in its README. The order in which positions are committed is saved for every sample (cheap, and phase 3 needs it).

**Sampling temperature.** Qwen runs at its vendor defaults. For each diffusion model, the temperature is picked on the dev set from {0.3, 0.5, 0.7, 1.0} as the value whose mean number of distinct outputs per 8 samples is closest to Qwen's. Reason: 8 samples only mean something if they differ, and comparing a near-greedy sampler with a diverse one confounds quality with diversity. Distinct counts are reported with the results.

**Seeds.** One seed per (model, item, direction), derived from a base seed and the item ID; the 8 samples are drawn in one batch from that generator.

**Answer extraction (same rule for all models).** Decode up to the first end token; take the first non-empty line; strip a leading label such as “下句：” and surrounding quotes or punctuation; the answer is what remains. Its length is its count of 汉字. Unit-tested.

### 9. Checker definitions (planned, to be frozen before the full run)

Pass/fail rules:

1. **Length**: the generated line has the same number of 汉字 as the given line, and contains nothing else.
2. **End tones**: the first line ends 仄, the second ends 平.
   - Modern: pypinyin reading in context; tones 1–2 are 平, 3–4 are 仄. A neutral-tone reading falls back to the character's citation tone.
   - Classical: 平水韻 table built from the Wikisource text of 《平水韻》 (revision 8264804), mapped to simplified characters. A simplified character whose traditional sources fall in both 平 and 仄 groups counts as either. Characters missing from the table fall back to the modern reading and are counted.
   - Both are reported. The all-rules figure uses the modern tones; a second all-rules figure uses the classical ones.
3. **No repeated characters** between the two lines, except where a character sits at the same positions in both.

Scores:

4. **Parallelism**: both lines are segmented and tagged; each character takes its word's coarse word class; the score is the share of positions where the two lines agree. Word-boundary agreement is reported beside it. Tagger: jieba.posseg unless a better free tagger separates true couplets from mismatched pairs more clearly on the dev set's reference couplets (decided before any generation is scored).
5. **Even-position tone opposition** (optional rule): share of even positions where the two lines have opposite tones. Reported, not part of all-rules.
6. **Relevance**: cosine similarity between bge-m3 embeddings of the generated line and the summary. Only for items with a summary.
7. **Match to the reference**: exact match and character overlap with the original line. This is the memorization meter: high on the recall set, near zero on the unseen set if the test works.

All-rules pass = rules 1, 2 and 3 together. The reference couplets are scored with the same checker as the ceiling.

Statistics: rates are per sample, with bootstrap intervals resampled by item (the 8 samples of an item are not independent). The headline figure is the unseen-minus-recall gap per model and direction, on matched conditions (with summary against with summary).

### 10. Git

The repo is initialized on `main`. No git identity is configured on this machine, so nothing is committed yet. Frozen files will also be listed with SHA-256 hashes in `FROZEN.sha256`, so the freeze does not depend on commits.

### 11. Unseen set built and frozen (version 1)

- File: `data/unseen/huohou.jsonl`, 43 items, one per line: `id`, chapter, title, `first`, `second`, `summary`, `summary_by` ("Claude"), whether the summary was edited in review, and the summariser's note.
- SHA-256: `307fa8861046aefcd431b2571ae1c5136ed5988453915955fa11b8c0e16e52c9` (also in `FROZEN.sha256`).
- Built with a one-off Perl script, because Python is not yet available. `scripts/build_unseen.py` will rebuild the couplet fields from the source clone and must reproduce this file exactly; if it does not, that is a bug to record here.

Review of the 43 summaries by the main session:

- Automatic check: all 43 have 2–3 sentences and 84–109 汉字. The only runs of three or more characters shared with a 回目 are personal names (伊内丝, 萨凡纳, 莉迪娅, 阿丝玛, 蒲嬢嬢, 以斯拉, 布雷登, 徐好翔, 路易斯, 优素福). No other exceptions were needed.
- Fact spot-check against the chapter text, by text search: 17 details from 15 chapters (figures, names, objects). All were found in the text as the summaries state them, except that one figure was written 二百六十 where the book has 两百六十 (edited, below). Some of these checks are weak (a name merely occurring in the chapter). This is a spot-check, not a full verification; the author will spot-check more.
- Three edits in review, each to wording only:
  - ch. 2: “茶铺耍起替美国人上的远程夜班” → “耍起茶铺替美国人上的远程夜班” (耍起 is the tea house's name; the original order read as a verb).
  - ch. 4: “科琳罚他撒谎，” → “科琳为这句谎话罚他，” (the original could be read as "made him lie as a punishment").
  - ch. 19: “二百六十毫秒” → “两百六十毫秒” (the book's own wording).
- Raw, unedited agent output stays in `data/unseen/summaries_raw/`.

Summarisers' notes worth the author's eye (the summaries follow the chapter text, not the 回目, where the two differ):

- ch. 25 “坐到底”: in the text 唐蕾 says she will not ride to the terminus.
- ch. 26 “隔屏”: the negotiation is a phone call on speaker, not a video call.
- ch. 30 “连夜”: the medicine is made up between 3 and 5 in the afternoon. “头回”: the text does not say it is the first time.
- ch. 31 “五点”: the light goes on at half past five.
- ch. 17: both lines of the 回目 point to the same thread, not two.
- ch. 40: the 回目 says 优素福; the chapter text calls him 迦勒 throughout. The summary gives both names.
- ch. 33: that 张磊 is 张一帆's father is established in other chapters, not in this one.
- ch. 12: “吻别” in the summary is inferred from the line after the three blank lines.

### 12. Code written but not yet run

`src/dfl/text.py` (counting 汉字) and `src/dfl/extract.py` (answer extraction) are written. Nothing has been executed: there is no Python yet. The checker, its unit tests, the set builders and the generation code are deliberately not written blind; they come once tests can run.

### 13. Status at the end of the first session

Done: machine check, model verification, unseen set with frozen summaries, sources located for the recall set, the dev set and the 平水韻 table, `CLAUDE.md`, this log.

Blocked: everything that needs Python (environment, model downloads, smoke test). Waiting for the author's go-ahead to install a Python toolchain. Checkpoint 1 (smoke test of three models on 3 items) has **not** been reached.

---

## 2026-11-19, second session (after the author's go-ahead: "Yes, install it")

### 14. Environment

- `uv` 0.13.9, one binary from its official GitHub release, in `../_tools/uv/` (outside the repo, not on PATH).
- Python 3.11.18, downloaded by uv into the user profile. uv's own step that links the minor version failed on this machine; pointing uv at the downloaded interpreter directly worked.
- `uv sync` built `.venv` from `pyproject.toml`; `uv.lock` records every version. Key ones: torch 2.5.1+cu124 and transformers 4.46.2 (the two versions Dream's README asks for), accelerate 1.2.1, bitsandbytes 0.50.2, sentence-transformers 3.4.1, pypinyin 0.55.0, jieba 0.42.1, opencc-python-reimplemented.
- Run everything with `PYTHONUTF8=1`: the Windows console code page cannot print Chinese otherwise.

### 15. Model downloads

- The Hugging Face cache could not be used: it needs symlinks, and Windows refused the privilege (WinError 1314). Decided: download into a plain folder, `../_models/<key>/`, with `scripts/download_models.py`, pinned to the registry commits. Each finished model gets a `dfl_revision.json`; the loader refuses a folder whose recorded commit differs from the registry.
- Downloaded: bge-m3 2.30 GB, Qwen 15.24 GB, Dream 15.24 GB, LLaDA 16.04 GB. About 11 minutes in all.
- Loading from a local folder also means the custom code that runs is the code at the pinned commit, read in entry 3.

### 16. Precision: decided, bf16 for all three

Plan 1 of entry 2 works. All three models run in bf16 with the input embedding table on the CPU and everything else on the GPU. No quantization.

| Model | GPU after load | Peak during generation |
|---|---|---|
| Qwen2.5-7B-Instruct | 13.21 GiB | 13.6 GiB |
| Dream-v0-Instruct-7B | 13.21 GiB | 13.8 GiB (4 per batch) |
| LLaDA-8B-Instruct | 13.97 GiB | 14.4 GiB |

One problem found and fixed. With 8 samples per batch and prompts of varying length, PyTorch's cached memory for Dream grew record by record (14.7, 15.7, 16.7, 17.8 GiB reserved) and spilled past the 16 GB card into shared memory: a record went from 5 s to 52 s. Outputs were unaffected. Decided, for all three models alike: the cache is emptied before every record, and the 8 samples go through in two batches of 4. The second batch gets its own seed, derived from the item seed and recorded.

For LLaDA the output head is run on the canvas positions only (a hook slices its input). The numbers at those positions are the same; it avoids building vocabulary-sized logits for the 200-token prompt.

### 17. Checker written, with unit tests

`src/dfl/`: `text.py`, `extract.py`, `tones.py`, `parallel.py`, `checker.py`, `scoring.py`. 103 unit tests in `tests/`, all passing. Decisions made while writing it, beyond entry 9:

- **The tone rule is judged on the generated line only.** A second line must end 平, a first line must end 仄. The given line is never blamed: some original lines break the convention, and a model cannot fix the line it was handed.
- **A plain copy of the given line fails all-rules.** It would otherwise pass rule 3 through the same-position exception. Rule 3 itself is reported as the author stated it.
- **Same-position exception**: a shared character is allowed only if it occupies the same set of positions in both lines.
- Rules 2 and 3 are evaluated on the 汉字 of the answer, so a line that fails length because of stray punctuation is still told whether its tone was right. An empty answer fails every rule.
- **Extraction**, refined on the Qwen and Dream smoke outputs (dev data): lines carrying the label of the given line are skipped; a line carrying the label of the requested line is preferred; a label alone on its line means the answer is the next line. Full rule in `extract.py`.
- Neutral-tone readings fall back to the character's first toned reading.

### 18. 平水韵 table built

`data/lexicon/pingshui.json`, from Wikisource 《平水韻》 revision 8264804 (106 rhyme groups, 9,000 characters in the main lists and 1,686 in the 【詞】 additions). 10,872 keys after adding simplified forms; 2,119 have an 入声 reading. SHA-256 `9e38ea5d…5cba`.

Limits: 1,167 keys sit in both a 平 and a 仄 group (看, 中, 张, 烧 and so on) and pass either requirement, which makes the classical check lenient. A character absent from the table (碗, 嬢, for example) falls back to its modern tone; this happens for 7% of line-final characters in the unseen set and 1% in the recall set.

### 19. Recall set and dev set built

- `data/recall/recall.jsonl`: **580 couplets**: 红楼梦 120, 水浒传 120, 三国演义 120, 西游记 100, 儒林外史 56, 海上花列传 64. Wikisource revisions are in `data/recall/sources.json`; the build reads exactly those revisions. SHA-256 `a4ead9c2…d948`. Not yet frozen.
- Line lengths: 8 characters in 302 couplets, 7 in 259, and a few of 4, 5, 6 and 9. No couplet has lines of unequal length.
- Markup handled: `{{另|A|B}}` and `{{參|A|note}}` become A; `-{A}-` becomes A.
- Conversion: OpenCC `t2s`. Then, so the lines read as common simplified editions print them: 餽→馈, 擡→抬, 渰→淹, 姪→侄, 雠→仇, 慇→殷, 懃→勤, 倣→仿, 珮→佩, 祇怕→只怕. A character whose formal simplification lies outside the Basic Multilingual Plane (綝, 駪) is kept as in the source. The 72 lines containing a character that converts ambiguously were read by eye; none needed a further fix. 17 lines keep a rare character that is part of a name or a set term (李傕, 张郃, 猇亭, 姽婳 and the like).
- `data/dev/dev.jsonl`: 100 couplets of 《镜花缘》. Eight have a summary, written by Claude from memory and unchecked; they exist only to exercise the with-summary prompt. Smoke and pilot item IDs are fixed in `data/dev/splits.json`.

### 20. Generation code

`src/dfl/models.py`, `generate.py`, `prompts.py`, `registry.py`; `scripts/run_generation.py` writes a run file per model with a header (model commit, precision, versions, settings) and one record per item, direction and condition (prompt, seed, effective settings, every sample's text and token ids).

- **Qwen**: `model.generate` at the model's own defaults (temperature 0.7, top_p 0.8, top_k 20, repetition penalty 1.05), up to 64 new tokens. The budget is generous on purpose, so a compliant reply is never cut.
- **LLaDA**: a port of the authors' reference sampler: Gumbel-max in float64, low-confidence remasking, no classifier-free guidance. Added: commit-order recording, a choice of unmasking order (confidence, random, left-to-right, right-to-left) and fixed anchors in the canvas. The last two are for phases 3 and 2 and are not used in phase 1.
- **Dream**: the authors' own `diffusion_generate`, called as their README shows (`alg="entropy"`, `alg_temp=0`, top_p 0.95). With steps equal to the canvas length, their schedule commits nothing at the first step and two tokens at the last; that is their behaviour and is left alone.
- Chat formatting: each tokenizer's own template, one user message, no system prompt added (Qwen's template inserts its default one).
- A diffusion reply is the canvas up to its first end token. Anything after it is dropped and flagged.
- Prompt wording, version `v1`, is in `prompts.py`. It states all the checked rules and says nothing about the tone of the given line.

### 21. Smoke test (checkpoint 1)

Three dev items (jhy-003, jhy-007, jhy-033), both directions, with and without summary, 8 samples each: 96 samples per model. Default settings of each model. Run files in `runs/smoke/`, scored rows in `results/smoke_scored.jsonl`.

| Model | Seconds per 8 samples (bare / with summary) | Length rule | Distinct outputs per 8 |
|---|---|---|---|
| Qwen | 0.6 / 0.7 | 91% | 5.8 |
| Dream | 3.2 / about 5 | 42% | 1.6 |
| LLaDA | 3.3 / 5.1 | 100% | 1.0 |

These are 3 items; the rates mean "the pipeline works", nothing more. What the smoke test did show:

- At their default temperatures the diffusion models return almost the same line 8 times (LLaDA is deterministic at temperature 0). The temperature calibration planned in entry 8 is needed.
- Dream prefixes “下句：” to bare replies (3 of its canvas tokens) and, given a summary, tends to write a clause with a comma instead of a 回目 line. LLaDA and Qwen write the bare line.
- Dream fills its canvas nearly left to right; the saved commit order shows it.
- Canvas slack of 4 is the least that holds a labelled line plus an end token. Under all three tokenizers a reference line takes at most 3 tokens more than its character count, and usually the same or fewer.
- No model reproduced a 《镜花缘》 line exactly. The dev novel does not seem to be memorized, which suits a dev set.

### 22. Ceiling check on the reference couplets, and two findings

The checker was run on the original couplets of all three sets (no generation involved).

| Set | First line ends 仄 (modern / classical) | Second line ends 平 | No repeat | Both lines pass all rules (modern / classical) |
|---|---|---|---|---|
| 《火候》, 43 | 98% / 98% | 100% / 100% | 93% | 91% / 91% |
| Recall, 580 | 46% / 60% | 60% / 59% | 97% | 43% / 52% |
| Dev (镜花缘), 100 | 55% / 63% | 63% / 64% | 100% | 55% / 60% |

**Finding 1: the classic novels do not keep the end-tone rule.** Only about half of their 回目 end 仄 then 平. The author's own 回目 keep it almost without exception. Consequence for the design: on the recall set, a model that recalls the original line will fail the tone rule about half the time, while the prompt asks it to keep the rule. So on recall items the all-rules rate measures obedience to the stated rule against the pull of the memorized line; it is not a ceiling the models should reach by recall. The per-rule rates and the exact-match rate have to be read together. This goes into `results.md`.

**Finding 2: the jieba parallelism score is weak.** True couplets average 0.63 on the recall set; randomly mismatched pairs of the same length average 0.51. The true couplet beats its mismatched partner in only 59% of pairs (18% ties). The score separates in aggregate and is close to noise per item. Open: try a better free tagger, or a score that does not depend on segmenting short classical lines, before the metric is frozen.

### 23. Open at checkpoint 1

- Recall summaries: how many chapters to summarise from the text (entry 6). Proposed: 48, eight per novel, seeded.
- Parallelism score: improve before the freeze (entry 22).
- Diffusion temperature: calibrate on dev (entry 8), then pilot on the 10 dev items in `splits.json`.
- Still no git identity, so no commits. Hashes so far: unseen set in `FROZEN.sha256`; recall, dev, 平水韵 table above; `uv.lock` `e40f7700…aa3a`.

---

## 2026-11-19, third session (after checkpoint 1; the author chose "Yes, 48 chapters")

### 24. Recall summaries: 48 chapters, from the text

- **Sample**: 8 chapters per novel, drawn with seed 20261119 by `scripts/fetch_recall_chapters.py`. The IDs and the Wikisource revision of every chapter page are in `data/recall/summary_sample.json`.
- **Text**: each chapter page was fetched, stripped of wiki markup and converted to simplified characters. The texts are working material in `data/raw/recall_chapters/` (not in git). Lengths run from 3,100 to 9,900 汉字.
- **Method**: six Claude subagents, one per novel, under the same seven rules as the unseen set. Rule 6 was reworded for old vernacular and (海上花列传) Suzhou-dialect dialogue. One instruction was added: write only what this chapter's text contains, not what you remember of the novel.
- **Check** (`scripts/merge_recall_summaries.py`): all 48 have 2–3 sentences and 89–108 汉字. The only runs of three or more characters shared with a 回目 are 19 names of people and places (诸葛亮, 鲁智深, 野猪林, 高老庄 and so on), listed in the script. No edits were made in review.
- **Fact spot-check** by text search: 18 details in 15 chapters, all found as stated.
- Merged into `data/recall/recall.jsonl` (SHA-256 `b45edbc2…a58d5c`); raw agent output in `data/recall/summaries_raw/`.

**A property of the classics that the unseen set does not share.** In several chapters the text does not contain the event one line of the 回目 names: the chapter breaks off first and the event opens the next chapter. The summarisers reported it and kept to the text. Each item carries `summary_covers`:

| Value | Items |
|---|---|
| `one_line_missing` | shz-008 (鲁智深 does not appear), shz-028 (快活林 is not mentioned), rlws-018 (潘三 does not appear), hshlz-056 (the plot named in the first line is not shown) |
| `one_line_weak` | shz-029, rlws-012 (the event is begun, not finished) |
| `both` | the other 42 |

All 43 unseen summaries cover both events. Decided: the recall with-summary results are reported on all 48 and again on the 42 fully covered ones. The summaries are not patched with text from the next chapter, because "the summary describes this chapter's text" is the rule both sets were built under.

### 25. Parallelism score: tagger chosen

Test (`scripts/compare_taggers.py`): for each reference couplet in the recall and dev sets, score the true pair against a mismatched pair of the same length drawn with a fixed seed. No model output is involved, and the unseen set was not used to choose.

| Tagger | Recall: true vs mismatched | True pair wins (ties count half) | Dev: true vs mismatched | Wins |
|---|---|---|---|---|
| jieba.posseg | 0.63 vs 0.52 | 65% | 0.59 vs 0.48 | 66% |
| jieba dictionary, character by character | 0.47 vs 0.34 | 71% | 0.47 vs 0.37 | 66% |
| Stanza zh-hans | 0.65 vs 0.50 | 68% | 0.63 vs 0.50 | 66% |
| Stanza lzh (classical Chinese) | 0.64 vs 0.57 | 63% | 0.66 vs 0.58 | 65% |
| LTP small | 0.66 vs 0.49 | 72% | 0.68 vs 0.52 | 71% |
| LTP base | 0.74 vs 0.51 | 81% | 0.75 vs 0.54 | 78% |
| **LTP base2** | 0.75 vs 0.52 | 81% | 0.77 vs 0.55 | 80% |

**Decided: LTP base2** (`LTP/base2`, commit `70c57011…396f`), pinned in the registry. It and LTP base are level; base2 is smaller. On the unseen set, looked at only after choosing, it gives 0.79 for true pairs against 0.56 for mismatched ones. The score is still noisy for a single couplet (a mismatched pair averages about 0.5 because most positions are nouns), so it is reported as a mean over many samples, never as a pass or fail. jieba stays in the code as a fallback and for the unit tests.

Licence: LTP is free for research and personal use; commercial use needs a licence from its authors (HIT). Stanza was installed for the comparison and is not used in scoring.

### 26. Statistics

`src/dfl/stats.py`, with unit tests: every interval is a bootstrap over items (4,000 draws, fixed seed), never over single samples, because the 8 samples of an item are not independent. Three comparisons: unseen minus recall (two sets, resampled separately); one model minus another on the same items (resampled together); and the difference of those differences between sets. `scripts/report.py` builds the tables.

### 27. Temperature calibration (dev)

Run by `scripts/calibrate.py` on the 20 dev items of the `calibration` split (none of them in the pilot), both directions, no summary, under the rule fixed in entry 8. Results are entered below when the run finishes.

**Temperature results** (40 records per row: 20 items, two directions, 8 samples each; scored with the checker):

| Model | Temperature | Distinct per 8 | Length | End tone | No repeat | All rules | Seconds per record |
|---|---|---|---|---|---|---|---|
| Qwen | default (0.7) | 6.85 | 88% | 51% | 93% | 44% | 0.8 |
| Dream | 0.3 | 3.30 | 62% | 48% | 64% | 22% | 3.3 |
| Dream | **0.5** | 6.60 | 42% | 58% | 54% | 15% | 3.3 |
| Dream | 0.7 | 7.62 | 23% | 53% | 57% | 8% | 3.3 |
| Dream | 1.0 | 7.97 | 26% | 48% | 62% | 12% | 3.3 |
| LLaDA | 0.3 | 4.40 | 90% | 59% | 76% | 46% | 4.0 |
| LLaDA | 0.5 | 6.33 | 86% | 58% | 75% | 44% | 4.0 |
| LLaDA | **0.7** | 7.22 | 83% | 59% | 76% | 42% | 4.0 |
| LLaDA | 1.0 | 7.92 | 65% | 55% | 79% | 30% | 4.0 |

**Chosen by the rule of entry 8** (distinct count closest to Qwen's 6.85): **Dream 0.5** (6.60) and **LLaDA 0.7** (7.22; LLaDA at 0.5 gives 6.33, which is further off).

What the table shows beyond the choice: LLaDA barely loses quality as temperature rises to 0.7; Dream loses it fast. So the matched-diversity comparison is hard on Dream, and a comparison at vendor defaults would flatter both diffusion models on rule-keeping while giving 1 to 3 distinct lines per 8. Proposed for the full run (decision for the author at checkpoint 2): run the diffusion models in **two arms**, matched diversity (primary, as planned) and vendor default (Dream 0.2, LLaDA 0), so the result does not hang on this one choice.

**Canvas slack, second step (rule written before the slack runs).** A look at the finished Dream runs showed its replies are never cut by the canvas (an end token is always present), but 82% of its bare replies carry a “下句：” label, which takes 3 of the 4 slack tokens. A labelled line that also ends in “。” needs L + 5 tokens and has L + 4; Dream then tends to drop a character (18% of its replies are one character short, against 8% for Qwen). So slack 4 may be costing Dream length passes. Rule: one slack for both diffusion models, the value from {4, 6, 8} with the highest all-rules pass rate averaged over the two models at their chosen temperatures, on the same 20 calibration items; a tie goes to the smaller canvas. Qwen is not affected: its budget is 64 tokens.

**Canvas slack results** (same 20 items, chosen temperatures):

| Slack | Dream: length / all rules | LLaDA: length / all rules | Mean all rules |
|---|---|---|---|
| 4 | 42% / 15% | 83% / 42% | 28.4% |
| **6** | 40% / 18% | 88% / 43% | 30.5% |
| 8 | 23% / 11% | 87% / 43% | 27.0% |

**Chosen by the rule: slack 6** (canvas = required characters + 6 tokens). The differences are small, and the guess that slack 4 was squeezing Dream's length is not borne out: Dream's length rate does not rise with more room (its gain at 6 comes from fewer repeated characters), and at 8 it gets worse. Dream's short lines are miscounts, not truncation. LLaDA is indifferent to slack. Distinct answers per 8 at slack 6: Dream 6.83, LLaDA 7.22, against Qwen's 6.85, so the temperatures still match.

**Settings written to `config/phase1.json`** (five arms): Qwen at vendor defaults; Dream 0.5 and LLaDA 0.7 (matched diversity); Dream 0.2 and LLaDA 0 (vendor defaults; LLaDA at 0 is deterministic, so it gets one sample per item instead of eight). Canvas slack 6, steps equal to canvas length, 4 samples per batch, base seed 20261119, bf16.

### 27b. Pilot (dev), and a prompt-layout check it prompted

The pilot (`scripts/run_plan.py pilot`: five arms, the 10 pilot items, with summary and bare; 12 minutes) ran without error and scored cleanly. Tables in `results/pilot_report.md`. It is dev data from one novel, so it shows behaviour, not the result.

What it showed that matters for the design: **given a summary, Dream almost never writes one line.** 77% of its with-summary replies are two short clauses joined by a comma (“花仙子来贺，众仙看作稀奇。”), at both temperatures, and some contain stray Latin tokens. Its length pass rate with a summary is 0–12%. Without a summary it writes a labelled single line. LLaDA and Qwen write single lines in both conditions.

That could be the model or it could be the prompt layout (the rules come after the material in v1, and the reply starts cold). To keep the result from hanging on layout, one alternative is tested on dev for all three primary arms: **v2** has exactly the same sentences in a different order (task, rules, closing instruction, then the material) and ends with the empty label of the line to be written (“下句：”).

**Rule, written before any v2 output was generated:** adopt v2 only if its all-rules pass rate, averaged over the three primary arms, is higher than v1's on the same dev items (the 10 pilot items with summary and bare, plus the 20 calibration items bare, at the frozen sampling settings). Otherwise keep v1. One alternative only; no further prompt variants will be tried.

**Prompt layout results** (`scripts/compare_prompts.py`; 128 samples per arm with summary, 480 bare):

| Layout | Arm | With summary: length / all rules | Bare: length / all rules | All rules, all samples |
|---|---|---|---|---|
| v1 | Qwen | 69% / 42% | 88% / 44% | 43.6% |
| v1 | Dream | 3% / 0% | 36% / 15% | 12.2% |
| v1 | LLaDA | 83% / 31% | 88% / 40% | 38.2% |
| v2 | Qwen | 85% / 52% | 88% / 50% | 50.5% |
| v2 | Dream | 11% / 7% | 23% / 13% | 12.0% |
| v2 | LLaDA | 86% / 38% | 86% / 40% | 39.3% |

Mean over the three arms: v1 31.3%, v2 33.9%. **Chosen by the rule: v2.** `config/phase1.json` and the default in `prompts.py` now say v2; v1 stays in the code for the record.

What the check settled: the layout was part of Dream's with-summary problem but not most of it. Under v2 the share of its with-summary replies that are clean single lines rose from 22% to 79%, yet only 11% have the right length. Qwen gained the most from v2 (its with-summary length rate went from 69% to 85%). So v2 does not favour the diffusion models; it was adopted because the rule said so.

The first pilot is kept as `runs/pilot_v1/` and `results/pilot_v1_*`. The pilot was run again with the final settings; its tables are in `results/pilot_report.md`.

### 27c. Final pilot (checkpoint 2) and a reproducibility check

Pilot with the final settings: five arms, 10 dev items, with summary (8 items) and bare (10), both directions. 13 minutes. Pooled over conditions and directions:

| Arm | Samples | Length | End tone | No repeat | All rules | Parallelism | Relevance (with summary) | Seconds per 8 samples |
|---|---|---|---|---|---|---|---|---|
| Qwen (default) | 288 | 85% | 58% | 93% | 51% | 0.73 | 0.51 | 1.0 |
| LLaDA 0.7 | 288 | 85% | 51% | 86% | 38% | 0.70 | 0.57 | 5.5 |
| Dream 0.5 | 288 | 16% | 50% | 75% | 8% | 0.24 | 0.51 | 4.8 |
| LLaDA 0 (default) | 36 | 92% | 50% | 86% | 44% | 0.79 | 0.58 | 1.0 (one sample) |
| Dream 0.2 (default) | 288 | 18% | 43% | 79% | 6% | 0.36 | 0.55 | 4.8 |

Per condition and direction, with intervals: `results/pilot_report.md`. This is 10 items of one dev novel; it is not the result. What it is good for:

- The pipeline runs end to end for all five arms, and every sample was scored.
- Dream's length failure holds at both temperatures; it is not an artefact of the matched-diversity setting.
- LLaDA and Qwen are in the same range; the pilot cannot rank them.
- No original 《镜花缘》 line was reproduced by any arm (exact match 0%).
- In the reverse direction every arm ends the first line in 仄 only 20–56% of the time (modern tones, by condition): the hardest rule for all of them.

**Reproducibility.** The three primary arms were generated twice on the pilot items in separate processes (once in the prompt-layout check, once in the final pilot), with the same seeds and settings. All 864 samples are token-for-token identical. A run can be regenerated exactly on this machine and environment.

### 29. Proposed freeze (awaiting the author at checkpoint 2)

Nothing has been generated on the unseen or recall sets. On the author's go-ahead, `scripts/freeze.py --write` records the SHA-256 of every file below and the full run starts; after that, a change to any of them needs a dated entry here.

- **Sets**: unseen v1 (43), recall (580, of which 48 with summaries), dev and its splits; the 平水韵 table.
- **Settings** (`config/phase1.json`): prompt layout v2; canvas of required characters + 6 tokens; one denoising step per canvas token; 8 samples per item in two batches of 4; base seed 20261119; bf16. Arms: Qwen at vendor defaults; Dream 0.5 and LLaDA 0.7 (primary, matched diversity); Dream 0.2 and LLaDA 0 (vendor defaults; a robustness arm the author may drop).
- **Code**: prompt, extraction, checker, tones, parallelism (LTP base2), scoring, statistics, generation, loading, registry, run and report scripts, grading-sheet builder; `uv.lock`.
- **Plan**: entry 28.

Estimated GPU time for the full run, from the pilot's timings: Qwen 25 minutes, Dream 1.7 hours per arm, LLaDA 2 hours for the primary arm and 25 minutes for the deterministic one. Three primary arms: about 4.2 hours. All five: about 6.3 hours.

## 2026-11-21

### 30. The author graded the dry-run sheet (not part of any result)

On 2026-11-19 a test of `make_grading_sheet.py` built a 9-line sheet from the **smoke-test** outputs (3 dev items of 《镜花缘》, old settings: prompt v1, vendor-default temperatures, slack 4) and opened it in the app's browser pane to check that it rendered. The file was deleted afterwards, but the pane kept showing it. The author took it for the real sheet and graded all 9 lines. The mistake was mine: a test page was left in front of the author.

What was done: the 9 scores were read from the open page; the dry-run sheet was regenerated from the same inputs and seed; every entry code and line matched what the author graded. The scores are saved in `results/grading_dryrun/grades.json` and unblinded with `scripts/unblind.py`, which doubles as an end-to-end test of the grading and unblinding steps. Result in `results/grading_dryrun/grades_report.md`: Qwen 4, 4, 4; LLaDA 3, 2, 4; Dream 2, 1, 1.

These 9 grades are **not** part of the study: dev items, superseded settings, three cells. They say only that the sheet works and that the author's scale separates broken lines (Dream's two-clause replies got 1) from usable ones. One observation worth keeping for the real grading: the author gave 4 to a LLaDA line that fails the end-tone rule (红颜女穿耳遭痛: a second line must end 平, and 痛 is 仄), so grade and rules do not coincide.

The real sheet (51 lines from the unseen set) is built only after the full run, which has not started.

### 31. Freeze and start of the full run

The author answered the checkpoint 2 question with "Yes, all five arms".

- Pre-freeze checks: 138 unit tests pass; the unseen set verifies against the source clone; GPU idle; 2.1 TB free.
- Entries 28 and 29 were copied verbatim into `PLAN_FROZEN.md`, so the plan is hashed with everything else.
- `scripts/freeze.py --write` recorded 28 files in `FROZEN.sha256` (sets, settings, prompt, extraction, checker, tones, parallelism, scoring, statistics, generation, loading, registry, run, report, grading and unblinding scripts, the plan, `uv.lock`). `scripts/unblind.py` was added to the frozen list at this point. From here on, a change to any listed file needs a dated entry here.
- `scripts/run_plan.py full` started on 2026-11-21 at about 18:27, all five arms in the order Qwen, Dream 0.5, LLaDA 0.7, Dream 0.2, LLaDA 0; each arm runs unseen with summary, unseen bare, recall with summary, recall bare. 1,428 records per arm. Log: `../_tools/full.log`; run files: `runs/full/`. Expected duration about 6.5 hours. An interruption loses at most the record in progress; rerunning the same command continues.

### 32. Before the first commit: local paths removed from run files

The author asked to commit and push. A scan before staging found no email address or personal detail in code, data or notes, but 42 finished run files recorded the item file as an absolute path, which contains this machine's Windows user folder. `scripts/strip_local_paths.py` rewrote that one header field to the repo-relative path (for example `data/dev/dev.jsonl`). Checked afterwards: every sample in the smoke, pilot v1 and pilot runs still matches the rows scored before the rewrite (2,664 of 2,664), and the pilot and prompt-layout runs are still token-identical (108 of 108 records).

`runs/full/` is being written by the full run, so it is left alone and not committed yet; the same rewrite is applied to it after the run ends. `scripts/run_generation.py` (frozen) still records absolute paths when given them; it is not changed during phase 1.

First commit `e28fd9f`, pushed on 2026-11-21 to the public repo https://github.com/yuanrui3/diffusion-fiction-lab (the author's choice: public, under yuanrui3, commits under the GitHub no-reply address). Created with the GitHub CLI (kept in `../_tools/gh/`), signed in by the author.

At the author's request the repo is licensed MIT (`LICENSE`, copyright holder given as the GitHub account yuanrui3). `NOTICE.md` lists the material derived from other sources (Wikisource, model licences), whose terms the MIT licence does not replace.

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

## 2026-11-22

### 33. Full run finished, scored, grading sheet built (checkpoint 3)

**Run.** `run_plan.py full` finished at about 00:38 after 371 minutes, exit 0. 20 run files, 1,428 records per arm, 7,140 records, 47,124 samples (8 per record; 1 for LLaDA 0). No errors, no duplicate records, no missing items. GPU time per arm: Qwen 20 min, Dream 0.5 101 min, LLaDA 0.7 120 min, Dream 0.2 101 min, LLaDA 0 23 min; peak memory 14.0 GiB.

**After the run, in this order.** The header rewrite of entry 32 was applied to all 20 files in `runs/full/` (no machine path left in `runs/` or `results/`). `freeze.py` then verified all 28 frozen files unchanged. Scored with relevance into `results/full_scored.jsonl` (47,124 rows); tables in `results/full_report.md`. No metric, set or setting was changed after the results were seen.

**Grading sheet.** `results/grading/grading_sheet.html`, built by the frozen script with the frozen seed: 51 entries from 17 cells of the unseen set, with-summary condition. 25 of the 51 entries came from the fallback of entry 28: in that cell the model had no sample among its 8 that kept all rules. Which models those are is in the key, which has not been opened. So about half the grades will say how good a model's ordinary line is, not its best rule-keeping one; the key records which is which.

**First reading, written before grading** (the full readout goes into `results.md` after the author has graded):

1. *Almost nothing is memorized.* On the recall set, bare, the original line is reproduced exactly by Qwen in 3% of forward samples and 0% of reverse ones, and by every diffusion arm in 0% of both. The recall set is therefore "famous couplets the models have probably seen", not "couplets they can recite". The unseen−recall gap must be read as new against familiar, not new against memorized. Question 2 of entry 28 (does diffusion do relatively better at recalling the earlier line, as in the LLaDA paper's poem test) cannot be answered here, because no arm recalls either line.
2. *The ceiling is low on the recall set.* The original classical couplets keep the modern end-tone rule only 60% of the time read forward and 46% read in reverse (table 4), so all-rules rates on the recall set cannot be compared with the unseen set's 91–93% ceiling as if the rules meant the same thing for both.
3. *Direction, not set, separates LLaDA from Qwen.* LLaDA 0.7 minus Qwen, all rules, forward: −18*, −18*, −8, −15* (unseen with summary, unseen bare, recall with summary, recall bare); reverse: +9, +13*, +9, +3*. The end-tone rule drives it: in reverse, Qwen ends the first line in 仄 only 20–35% of the time. This holds in both sets and both conditions, which entry 28 counts as stronger evidence than any single cell.
4. *Dream's result is a format failure.* Its length pass rate is 5–34% in every cell, as in the pilot; most of its replies are not a single line of the right length.

Not committed yet: the author is asked first (`CLAUDE.md`).

On 2026-11-23 the author answered "Yes, commit and push". Committed: `runs/full/`, `results/full_scored.jsonl`, `results/full_report.md`, the grading sheet and this entry. The key stays out of git (`.gitignore`) until grading is done.

## 2026-11-25

### 34. Grading done, grades unblinded, results written

**Grades.** The author graded the sheet in their own browser and exported `grades.json` (sheet `unseen-20261119`). It was copied from the browser's download folder into `results/grading/`. `freeze.py` verified all frozen files, then `scripts/unblind.py` read the key for the first time. Output: `results/grading/graded.jsonl` and `results/grading/grades_report.md`.

48 of the 51 lines were graded. Three were left blank: e11 (Qwen, a rule-passing line), e36 and e46 (both Dream fallbacks that mix in English words). Call made without asking: they count as missing, as `unblind.py` was written to do, and are not imputed. Scoring the two Dream lines as 1 would only widen Dream's gap.

| Model | Lines graded | Mean (1–5) | Scored 4 or 5 | From rule-passing samples |
|---|---|---|---|---|
| Qwen | 16 | 2.88 [2.25, 3.44] | 6 | 12 |
| Dream 0.5 | 15 | 1.60 [1.20, 2.07] | 1 | 3 |
| LLaDA 0.7 | 17 | 2.71 [2.29, 3.12] | 3 | 10 |

Paired by cell: LLaDA − Qwen −0.25 [−1.00, +0.50]; Dream − Qwen −1.43 [−2.21, −0.57]; LLaDA − Dream +1.20 [+0.67, +1.67].

**One analysis not in the frozen plan**, labelled as such in `results.md`: mean grade by whether the line passed all rules. Rule-passing lines averaged 2.88 (25 lines), fallbacks 1.91 (23 lines).

**Results.** `results/results.md` answers the question and lists what the numbers can and can't show. Its caveats include:
- Claude-assisted 回目 and Claude-written summaries.
- "Unseen" taken on the models' release dates, not verified.
- No memorization of the classics.
- Classical originals that don't keep the tone rule.
- One judge, 48 grades, half of them fallbacks.
- A fixed setup and the model sizes.

**Keys.** Both grading sheets (the dry run of entry 30 and this one) are now graded, so `.gitignore` no longer excludes their keys; `graded.jsonl` carries the same mapping anyway. A future sheet's key must be listed in `.gitignore` until its grading is done.

Phase 1 is complete. Phases 2 and 3 have not been started.

### 35. Draft release v0.2.0 for Zenodo

The author asked for a release, then for unattended work until 5 am without being asked to judge or decide. The repo already had a release, v0.1.0, made by the author on 2026-11-22 from a curated commit (`a6abbcd`, on no branch, tag only). That commit left out `data/unseen/` and the grading dry run, cut excerpts from both out of this log, and replaced unseen lines in the tests with made-up ones. Zenodo archives every published release (webhook present).

So v0.2.0 was built the same way, not from `main`:
- **Commit:** `7def136`, child of `a25b275`, on no branch; tag `v0.2.0` pushed.
- **Excluded:**
  - everything v0.1.0 left out;
  - `runs/full/*_unseen_*`;
  - the unseen rows of `results/full_scored.jsonl`, so the archived file has the 41,448 recall rows only;
  - `results/grading/`: sheet, grades, unblinded lines and key.
- **Rewritten for the archive:**
  - `results/results.md` describes, rather than quotes, the five model lines written about unseen chapters;
  - the scope note at the top of this log now covers v0.2.0;
  - `.zenodo.json` is at v0.2.0 with a new description.
- **Checks:**
  - A scan for the 86 回目 halves, 8-character runs of the summaries, and rare 3-character runs of the 回目 found none, apart from the names list v0.1.0 also kept.
  - All 138 tests pass in the archived tree.

**Gap in v0.1.0.** Its tests still contain four unseen 回目 halves (外婆借身炸米饺, 王鹏借壳给老邓洗澡 twice, 清汤面点两碗, 嬢嬢热抱屋头那个), so the published v0.1.0 archive holds them. v0.2.0 replaces them with made-up lines. Whether to do anything about v0.1.0 on Zenodo is the author's call.

The GitHub release is left as a **draft**. Publishing it creates a permanent Zenodo version, and the scope above is a reading of the author's v0.1.0 choices rather than an instruction, so the author publishes it. Nothing on `main` changed for the release.

## 2026-11-26

### 36. Theory exploration: post hoc analyses and probes (not part of the frozen plan)

The author asked for unattended work until 5 am exploring the theoretical side of the project, using only free APIs and never asking the author to judge. Everything below is **post hoc**, chosen after the results were seen. It lives in `explore/` and reads the frozen outputs without changing them; none of it is a pre-registered result. No new couplets were generated, and nothing of phases 2 or 3 was started.

**Literature.** Checked through the free arXiv API (titles, authors, dates and abstracts), so that every reference in `results/theory.md` is verified rather than recalled: any-order and masked diffusion theory, token-order planning, the reversal curse, next-token-prediction limits, counting and tokenization, and LLM tone compliance in Chinese poetry.

**Analyses of existing outputs** (`explore/analyse_existing.py`, CPU, output `explore/out/existing_data.md`):
1. *Line-final tone.* For each model and for the originals: the share of 平 endings by direction and by the given line's final tone (modern and classical). From these, a *rule effect* (direction contrast at fixed given tone) and an *imitation effect* (given-tone contrast at fixed direction). Only the recall set separates the two, because in 《火候》 direction and given tone always agree.
2. *What the originals keep.* Tone assignment (first line 仄, second 平) against opposition (the two lines end on different tones), novel by novel.
3. *Length.* Signed length error, the calibration runs' output length against canvas slack, and a confusion table of required against produced length (attraction toward 7 characters).
4. *Order.* From `commit_step` in `runs/full/`: whether the end padding is fixed before the line, the left-to-right correlation within the line, and when the last character is fixed.
5. *Copying.* The share of the answer's characters found in the summary and in the given line, and its correlation with the relevance score.

**Shuffled baseline for parallelism** (`explore/parallel_baseline.py`, LTP on the CPU): the frozen parallelism score of right-length answers, against the line actually given and against another item's given line of the same length and role. The first run passed raw answers instead of 汉字-only strings (the checker strips non-汉字 first), so Dream's answers with stray letters were dropped. It was corrected and rerun before any table was used.

**Probes** (GPU, one model per process, one forward pass per input, nothing sampled):
- `explore/probe_tones.py`:
  - *Probe K*, tone knowledge: “汉字“X”在普通话里读第几声？只回答一个数字。” for the 554 line-final characters with one unambiguous reading. The reply's first position is read over the digits 1–4.
  - *Probe S*, the final slot: the exact phase 1 prompt, followed by the original line minus its last character. The slot's mass on 平 is read among single-汉字 tokens, with the prompt as used and with rule 三 (the end tone) removed. Both directions, all 723 items.
- `explore/probe_length.py`, *Probe L*: on the fully masked phase 1 canvas, each position's probability of an end token, its entropy and its top probability. These give the length the model expects before the first step, and which position each sampler would fix first.

**Follow-up probes**, added after the first probe results raised questions:
- `probe_checks.py`, *K2*: Qwen put only 1% of its first-token mass on bare digits in probe K, so the tone question was asked again with the reply started as "第", reading digits and numerals together (78–86% of the mass lands there). It also runs a sanity check of the final-slot canvas on well-known couplets.
- `probe_S_masked_tail.py`: probe S with the tail masked rather than padded. That is the state Dream is actually in when it picks its last character.
- `probe_dream_order.py`: Dream's own entropy ranking at step 0, recomputed with its sampler's temperature, top-p and bf16 arithmetic, plus a check of its one-position shift.
- `rescore_opposition.py`: phase 1's all-rules rate with tone opposition in place of tone assignment, and the author's grades against copying from the summary.
- `relevance_vs_copy.py` (CPU): the relevance difference against Qwen, overall and within five bands of copy share.
- `copy_first.py` (CPU): commit rank of tokens copied from the summary against other tokens, overall and within each third of the line.
- `probe_rule_price.py`: how committed the final character is. On each model's own first sample (and on the original line), how often its favourite final character already has the class the rule wants, and how much less likely the best allowed character is when it doesn't.

**Main findings** (reading in `results/theory.md`; tables in `explore/out/`):
- *Tone.* No model follows the tone rule: the rule effect is at most +10 points out of 100, and deleting the rule from the prompt moves the final slot by at most 1 point. All three know Mandarin tones only moderately (K2 平/仄: 67–83%). All prefer 平 endings, and on 《火候》 that preference alone produces phase 1's "LLaDA better at first lines" pattern (an exact identity: pass difference = 2b − 1).
- *The classics' rule.* Classical 回目 keep tone opposition (86–100% by novel, classical tones) but not the assignment the checker tested (42–73%). Re-scored with opposition, the originals pass 77% on the recall set instead of 42%, while the models stay near 40%.
- *Order and length.*
  - LLaDA fixes its end padding first in 84–88% of samples, and on the fully masked canvas already expects the right length (−0.09 tokens on average).
  - Dream fills almost left to right because its sampler's truncated, bf16 entropy ties many positions at zero and breaks ties leftward; at temperature 1 its own beliefs would fix the padding first.
  - Neither beats Qwen on length. All drift toward 7-character lines.
- *Parallelism.* On right-length lines, Qwen and LLaDA align with the given line as much as the originals do above a shuffled baseline (+0.25 to +0.27 against +0.21 to +0.30). Dream aligns less.
- *Copying.* The diffusion models take 74–82% of their characters from the summary (Qwen 48%, originals 46%), and relevance rises with it. At equal copy share, LLaDA's relevance edge over Qwen on the unseen set halves (+0.061 → +0.036) but stays above zero; Dream's disappears. The author's grades fall with copying (ρ = −0.30 over 48 lines; within Qwen −0.64).
- *Dream's final slot is unreadable at temperature 1:* 13% of its mass falls on single 汉字 and 1.5% on the original; it generates usable text only because it samples at 0.2–0.5.
- *Copy first.* The adaptive samplers fix tokens copied from the summary before the line's other tokens. LLaDA 0.7: mean commit rank 0.43 against 0.68, and −0.22 within the same third of the line. Easy-first ordering builds each line around lifted characters, which helps explain the diffusion models' heavy copying.
- *The final word needs look-ahead after all.* `theory.md` first said an autoregressive model meets the end-tone rule at its last step, the best possible position. The price probe showed that was too strong. On Qwen's own first lines the favourite final character has the wrong tone 64–71% of the time, and the best allowed one is then a median 4.3–4.4 nats (about 75 times) less likely. The line is committed to its last word before the last step. `theory.md` §2, §3.1 and prediction 3 were revised: enforcing the tone from the first step, which a canvas allows, is the one place order should matter even with one line given.

Results and their reading: `results/theory.md`. Tables: `explore/out/`.

### 37. Correction to `results.md` section 3 (memorization)

The first version of `results/results.md` said that "no model recalls either line" of the classical 回目 and that the poem-reversal question could not be tested. Both were wrong. The frozen per-attempt exact-match rates round to 0–3%, and that hides what an item-level count shows (`explore/recall_direction.py`, post hoc):
- Qwen reproduced the original second line for 30 of 580 items, but the original first line for only 4 (Fisher p < 0.001).
- LLaDA 0.7 reproduced 1 and 4. The two direction profiles differ (Fisher p = 0.004).
- Dream reproduced none.

On the frozen partial-recall score (`ref_same_position`), averaged over all 580 items:
- Qwen: 9.2% forward, 4.2% reverse.
- LLaDA: 5.4% and 5.6%.
- LLaDA − Qwen: −3.8 [−5.2, −2.5] forward, +1.4 [+0.6, +2.2] reverse.

That is the LLaDA paper's reversal pattern, at a small scale. Section 3 of `results.md` was rewritten to say so, and the short answer and the "can show" list were adjusted, each marked with the date. The draft release v0.2.0 (entry 35) still carries the old wording. The fix reaches the archive only if it is committed and the draft rebuilt.

## 2026-11-27

### 38. Theory work committed

The author answered "Yes, commit and push" to the overnight work of entries 36–37. Committed together:
- `results/theory.md`;
- the `explore/` scripts and their outputs in `explore/out/`;
- the dated corrections and additions to `results/results.md`;
- this log;
- the `CLAUDE.md` note on `explore/`.

The draft release v0.2.0 (entry 35) is unchanged and still carries `results.md` as it was before entry 37.
