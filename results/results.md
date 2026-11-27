# Phase 1 results: 回目 couplets

**Question.** Do diffusion language models write Chinese 回目 couplets better than an autoregressive model of similar size, and does any advantage survive when the couplet is new rather than memorized?

**Short answer.** No. On the author's own new 回目, neither diffusion model wrote better couplets than Qwen2.5-7B-Instruct, by the rules or by the author's grades.
- LLaDA-8B comes close: the author's grades show no detectable difference from Qwen. By the rules it is worse at writing a second line and slightly better at writing a first line.
- Dream-v0-7B is clearly worse on every measure, mostly because it rarely produces a single line of the right length.
- The second half of the question can't really be asked. There was no diffusion advantage on the familiar classics either, and the models had memorized very few of them: Qwen about 30 of 580, mostly in one direction, and LLaDA a handful. *(Revised 2026-11-26; see the correction in section 3.)*

Everything below is from `results/full_report.md` (automatic scores), `results/grading/grades_report.md` (the author's grades) and `PROTOCOL.md` (every decision, dated). The plan was fixed before the run (`PLAN_FROZEN.md`, hashed in `FROZEN.sha256`).

## What was run

- **Models**:
  - Qwen2.5-7B-Instruct at its default sampling (autoregressive control).
  - LLaDA-8B-Instruct at temperature 0.7 and Dream-v0-Instruct-7B at 0.5. These temperatures were chosen on a dev novel so that the diffusion models give about as many distinct answers per 8 as Qwen.
  - Two robustness arms at the vendors' defaults: LLaDA at 0, which is deterministic (1 sample), and Dream at 0.2.
  - All in bf16 on one 16 GB GPU, with commits pinned.
- **Sets**:
  - *Unseen*: the 43 回目 of 《火候》.
  - *Recall*: 580 回目 from 红楼梦, 水浒传, 三国演义, 西游记, 儒林外史 and 海上花列传.
  - Each set was run with a 2–3 sentence chapter summary (43 unseen; 48 recall chapters sampled) and without one (bare).
- **Task**: given one line, write the other. *Forward* means writing the second line; *reverse* means writing the first. The prompt states the number of characters. Each model made 8 attempts per item and direction. For diffusion models the canvas is the required number of characters plus 6 tokens, with one denoising step per token.
- **Rules**, checked on the generated line only:
  - same number of 汉字 as the given line;
  - the right end tone (first line 仄, second line 平), by modern 普通话 tones and separately by 平水韵;
  - no repeated character except at matching positions;
  - not a copy of the given line.
  - "All rules" means length, modern end tone and no repeat together.
  - Also measured: part-of-speech parallelism (LTP tagger) and relevance to the summary (bge-m3 cosine).
- **Human grades**: the author graded 51 lines blind, 1–5. They came from 17 (chapter, direction) cells of the unseen set with summaries, one line per model per cell. Each line was drawn from that model's rule-passing attempts where there were any.

## 1. New couplets (unseen set, with summary)

| | Qwen | LLaDA 0.7 | Dream 0.5 | LLaDA 0 | Dream 0.2 |
|---|---|---|---|---|---|
| All rules, writing the second line | **52%** [40, 63] | 34% [24, 44] | 13% [5, 22] | 30% | 9% |
| All rules, writing the first line | 20% [12, 30] | **29%** [19, 40] | 12% [4, 22] | 21% | 0% |
| Right length (second / first line) | 66% / 73% | 68% / 66% | 28% / 34% | 67% / 65% | 21% / 10% |
| Parallelism, 0–1 (second / first line) | 0.46 / 0.55 | 0.47 / 0.49 | 0.24 / 0.24 | 0.46 / 0.50 | 0.17 / 0.05 |
| Relevance to the summary | 0.50 | 0.55–0.57 | 0.52 | 0.55–0.57 | 0.53–0.56 |
| **Author's grade**, mean of 1–5 | **2.88** [2.25, 3.44] | 2.71 [2.29, 3.12] | 1.60 [1.20, 2.07] | – | – |
| Lines graded 4 or 5 | 6 of 16 | 3 of 17 | 1 of 15 | – | – |

Brackets are 95% bootstrap intervals over items (over cells for the grades).

Paired on the same items, against Qwen:
- **All rules:**
  - LLaDA is 18 points worse at writing the second line [−32, −4], and 9 points better at writing the first [−4, +21], which is too small to call a difference.
  - Dream is 40 points worse at writing the second line and 8 points worse at the first (not detectable).
- **Author's grades:**
  - LLaDA − Qwen −0.25 [−1.00, +0.50]: no detectable difference.
  - Dream − Qwen −1.43 [−2.21, −0.57].
  - LLaDA − Dream +1.20 [+0.67, +1.67].
- **Relevance:** both diffusion models are slightly closer to the summary than Qwen (LLaDA +0.05 and +0.07, both with intervals above zero). The gain is small and the author's grades don't reflect it.

No model averages 3 ("acceptable") in the author's grades. Only one line got a 5: Qwen's 水房里凤侣传情.

## 2. The pattern that holds: direction, not set

LLaDA minus Qwen, all-rules pass rate, percentage points (* = 95% interval excludes zero):

| | Unseen, summary | Unseen, bare | Recall, summary | Recall, bare |
|---|---|---|---|---|
| Writing the second line | −18 * | −18 * | −8 | −15 * |
| Writing the first line | +9 | +13 * | +9 | +3 * |

The sign is the same in all four set-and-condition cells, which the plan counts as stronger evidence than any single cell. The end-tone rule drives it.
- **First line:** a first line must end on a 仄 character, and Qwen manages that only 20–35% of the time. LLaDA does 7–22 points better in every cell.
- **Second line:** LLaDA is worse than Qwen at ending it on 平.

The author's grades don't show the pattern: LLaDA 2.62 on first lines and 2.78 on second lines; Qwen 3.00 and 2.75. With 8–9 cells per direction, they couldn't have shown a difference of this size.

*Added 2026-11-26, post hoc (`results/theory.md`):* this pattern is not skill at the tone rule. Neither model follows the rule: whichever line they write, both prefer to end on 平, and Qwen more strongly. On 《火候》 the given line always ends the conventional way, so a model that ignores the rule but prefers 平 passes when writing second lines and fails when writing first lines. That model also does worse in reverse the more strongly it prefers 平.

## 3. Memorized couplets: very few

The recall set was meant to measure couplets the models know by heart. Without a summary, the original line was reproduced exactly:
- Qwen: 3% of attempts at second lines, 0% at first lines.
- Every diffusion arm: 0% in both directions (rounded).

Counted by item instead of by attempt, over both conditions:
- **Qwen** reproduced the original second line for 30 of the 580 classical items (147 attempts), but the original first line for only 4 (7 attempts).
- **LLaDA 0.7** reproduced 5 items: 1 when writing second lines (3 attempts) and 4 when writing first lines (15 attempts).
- **Dream:** none.

At this model size most classical 回目 are familiar text, not recited text. Two consequences:
- **The poem-reversal pattern shows up, but only in partial recall.** The question is whether diffusion recalls the earlier line better than an autoregressive model does, as in the LLaDA paper's poem test.
  - *Exact recall:* Qwen recalls the next line from the earlier one about 7 times as often as the reverse (30 items against 4, Fisher p < 0.001): the reversal curse. LLaDA shows no forward advantage, and the two models' direction profiles differ (Fisher p = 0.004 on the recalled items). But LLaDA's 5 items are too few to rest on.
  - *Partial recall,* the share of positions where the answer has the original character, over all 580 items:

    | | Forward | Reverse |
    |---|---|---|
    | Qwen | 9.2% | 4.2% |
    | LLaDA 0.7 | 5.4% | 5.6% |
    | LLaDA − Qwen | −3.8 [−5.2, −2.5] | +1.4 [+0.6, +2.2] |

    Qwen's recall is lopsided and LLaDA's is symmetric. LLaDA is worse than Qwen in the forward direction and better in reverse: the LLaDA paper's poem result, on 回目. The effect is a few characters per hundred.
  - *Dream:* almost nothing in either direction (1–1.5%), although it starts from Qwen2.5's weights. Converting the model to diffusion seems to have erased most of the verbatim memory, or its format failures hide it.
- **The unseen−recall gap means new against familiar,** not new against memorized.

*Correction, 2026-11-26.* The first version of this section said that "no model recalls either line" and that the poem-reversal question could not be tested. Both statements were wrong. The per-attempt rates round to 0–3%, which hid Qwen's 30 recalled items and the direction asymmetry above. The item counts and partial-recall figures come from a post hoc analysis (`explore/recall_direction.py`, `PROTOCOL.md` entry 37), not from the frozen tables; the partial-recall measure is the frozen `ref_same_position` score.

## 4. Does an advantage survive on new couplets?

There is no advantage on the recall set for anything to survive. The plan's interaction test compares "diffusion − Qwen on unseen" with the same difference on recall:
- **With summary:** no detectable change for either diffusion model in either direction.
- **Bare, writing the first line:** both diffusion models do relatively better on the unseen set (LLaDA +10 [+1, +19], Dream +9 [+2, +15]). This comes from Qwen getting worse on the new couplets (24% on recall, 15% on unseen), not from the diffusion models getting better (LLaDA 27% and 28%).

Within each model, the unseen−recall gap in all-rules pass rate is mostly within ±10 points and not detectable (table 2 of `full_report.md`).

## 5. Rules and grades

This analysis was not in the frozen plan.
- **Rule-passing lines scored higher:** they averaged 2.88 (25 lines); lines that broke at least one rule averaged 1.91 (23 lines).
- **But the rules don't decide the grade:**
  - Two Qwen lines that fail the rules got 4 from the author: 库内姐妹相争执 and 张帆看辫子忆先贤.
  - Two rule-passing lines got 1: Dream's 王鹏远程为老邓洗洗 and Qwen's 铁子初来学腰花切断.
- **A pass rate isn't a quality score.** The rules measure form; whether a line is a good 回目 is a separate question that only the grades address.

## What these numbers can show

- **The overall verdict:** under one fixed, recorded setup, two open 7–8B diffusion models do not write rule-keeping 回目 more often than a 7B autoregressive model, and the author did not prefer their lines.
- **A difference in kind, not in level:** LLaDA beats Qwen at ending a first line on 仄 and loses at ending a second line on 平, consistently across sets and conditions.
- **Dream's weakness is the format:** under this prompt and canvas it mostly writes two-clause sentences instead of one line of the given length. This held at both temperatures and in the pilot.
- **Memorization:** few classical 回目 are memorized by any of the three models. What Qwen does recall, it recalls forward only; LLaDA's weaker recall is the same in both directions (section 3, post hoc).

## What they cannot show

- **The 火候 回目 were drafted with Claude's help,** and the chapter summaries were written by Claude. The unseen references are not independent human couplets, and the summaries may sit closer to the 回目 than an independent reader's would. Relevance is measured against Claude's summaries.
- **"Unseen" is an assumption.** Qwen2.5 was released in September 2024, LLaDA in February 2025 and Dream (initialised from Qwen2.5) in April 2025. The 火候 回目 are taken to be newer than all three; this study did not verify when they were first published.
- **The rules mean different things on the two sets.** The classical originals keep the modern end-tone rule only 60% of the time (second lines) and 46% (first lines), against 100% and 98% for 火候. Pass rates on the recall set therefore can't be compared with the unseen set's as if they measured the same thing.
  - *Added 2026-11-26, post hoc (`results/theory.md` §3.2):* the classics keep a different rule. Their two lines almost always end on different tones (86–100% by novel, classical tones), but which line takes 平 is close to a coin toss.
  - Scored that way, the originals pass 77% on the recall set instead of 42%, while the models stay near 40%.
- **One judge and 48 grades.** The author graded 48 of 51 lines and left three blank: one of Qwen's and two of Dream's, both of which mix in English words. These count as missing, not as 1. The author also wrote the reference couplets, and grading was blind to the model, not to the author's own taste.
- **About half the graded lines were fallbacks.** In 25 of the 51 lines, that model had no rule-passing attempt among its 8 in that cell, so the line came from its other attempts: 12 of Dream's 15 graded lines, 7 of LLaDA's 17 and 4 of Qwen's 16. The grades therefore mix two questions: how good the model's best rule-keeping line is, and how good its ordinary line is.
- **Small samples:** 43 unseen items and 16–17 graded cells per model give wide intervals. Where this report says "no detectable difference", it means that, not "the same". Many comparisons were made with no correction for multiplicity.
- **The setup is fixed.** The prompt layout, canvas slack and temperatures were chosen on a dev novel (《镜花缘》) and frozen, together with each sampler's own unmasking rule. Other choices could change the result, especially for Dream. The vendor-default arms give the same picture.
- **The scale is small:** two diffusion models of 7–8B against one autoregressive model of 7B. This study says nothing about diffusion models in general, or at other sizes.
- **Recall summaries don't always fit.** Six of the 48 sampled recall chapters lack one of the two events in the chapter text available. Dropping them changes nothing (table 2, "fully covered").
- **The automatic scores are proxies.** Tones come from pypinyin and a 平水韵 table, with "both" readings counted as passing; parallelism comes from an automatic tagger. Both are unit-tested but imperfect.

## Reproducing

`config/phase1.json` holds every setting; `runs/full/` holds every prompt, seed, setting and raw output. On the pilot, regenerating the three primary arms in a separate process gave token-identical output (864 of 864 samples). The pipeline is listed in `CLAUDE.md`.
