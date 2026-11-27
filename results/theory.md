# Phase 1 in theory: what the 回目 task can show about diffusion

*Post hoc and exploratory, written 2026-11-26 after the results were in. Nothing frozen was changed and no new couplets were generated. Every analysis here was chosen after seeing the data, so it suggests rather than confirms (`PROTOCOL.md` entries 36–37). Scripts and tables: `explore/`.*

**In short.** Theory and the post hoc checks agree on one point. As phase 1 posed the task, one line given and the other written, the 回目 barely uses what makes a diffusion model different.
- **Most rules don't need a plan.** With the given line in view, every rule about the pairing becomes a rule about the line being written, one position at a time, and an autoregressive model handles that as well as any other order. Only two things need a little foresight: the length, and the final word, if the end tone is to be met gracefully.
- **Length does need a plan,** and there the samplers really differ. LLaDA settles the length before it writes a character. Dream writes almost left to right, an artefact of how its sampler breaks ties at low temperature. Neither beats Qwen, which simply counts.
- **The tone gap is not rule-following.** All three models know Mandarin tones to some degree, but stating the tone rule moves their choice of final character by at most 1 point. The forward/reverse asymmetry in phase 1 comes from a preference for 平 endings, which is stronger in Qwen.
- **The classical 回目 keep a different rule.** They contrast the two lines' final tones almost every time, but which line ends on 平 is close to a coin toss. The frozen checker tested the modern 对联 assignment, which 《火候》 keeps and the classics do not. Under the classics' own rule they pass 77%, not 42%, and the models stay near 40%.
- **The diffusion lines lean on the summary.** Both diffusion models build their lines out of the summary's characters, and fix the copied ones first. Copying accounts for about half of LLaDA's relevance edge and all of Dream's, and the author's grades fall as copying rises.
- **Memory follows the training objective.** Qwen recalls famous 回目 forward only; LLaDA's weaker recall is symmetric, and LLaDA is better than Qwen in reverse. Dream, built from Qwen's weights, has lost almost all of it.

The theory predicts the opposite where a later fixed point constrains earlier text, which is phase 2. Phase 1's null result is no evidence against diffusion there.

---

## 1. The task as a constraint problem

Write *y* for the line to be produced, *g* for the given line (*n* 汉字), *d* for the direction and *s* for the optional summary. The model samples *y* from p(*y* | prompt). The checker's rules define a set of acceptable lines, and the pass rate is the probability that a sample falls inside it. What matters for theory is which unknown positions each rule ties together:

| Rule | What it ties together | Unknowns involved |
|---|---|---|
| Length | the whole line: where it ends | all positions (a count) |
| End tone | the last character | one position, but "last" depends on length, and the last character depends on the last word |
| No repeat | each character against the given line and the line so far | one position against known text, plus the line itself |
| Parallelism (对仗) | position *i* of *y* against position *i* of *g* | one position against known text |
| Relevance | the line against the summary | the whole line, softly |

The given line is fully visible, so every rule about the pairing reduces to a rule on one position of *y* against known text. Inside *y*, the only real couplings are length, repeats within the line, and ordinary fluency. Fluency is what ties the last character to the words before it (§2). A Sudoku, the classic case where diffusion wins, is the opposite: its rules tie unknown cells to other unknown cells.

## 2. What theory predicts

**Masked diffusion is an any-order autoregressive model.** The absorbing-state objective that LLaDA and Dream train on is equivalent to an expected negative log-likelihood over all generation orders: it learns every conditional p(*x*ᵢ | *x*_S) (Ou et al. 2024; Hoogeboom et al. 2021; Shih et al. 2022). The order is chosen at inference. LLaDA's low-confidence remasking and Dream's entropy rule are greedy adaptive orders. An autoregressive model has one fixed order.

**Order matters only when an early decision depends on something decided later.** For an exact model every order gives the same distribution (the chain rule). In practice the difference is which conditionals are hard to learn or compute:
- Teacher-forced next-token training can fail outright on tasks that need lookahead (Bachmann & Nagarajan 2024).
- Autoregressive models struggle with "hard subgoals" placed early in the sequence; diffusion learns them (Ye et al. 2024, Countdown and Sudoku).
- An adaptive decoding order lets a masked model skip hard subproblems: on Sudoku, from under 7% to about 90% (Kim et al. 2025).

**Applied to one line given:**
- **End tone.** An autoregressive model can *check* this rule at its last step, with everything else visible. But checking is not meeting. By the last step the earlier characters have usually committed the line to its last word, and so to its tone. Meeting the rule then means choosing the final word ahead: a small look-ahead, about one word long. A canvas model can fix the final character first and write toward it. The rule is also hard if the model can't map characters to tones, or doesn't act on the instruction. (§3.1 measures how committed the models' own lines are by their last character.)
- **Parallelism and no repeat.** These look back at known text; no lookahead is needed.
- **Length.** This rule needs a real plan. An autoregressive model must stop after the *n*-th character while seeing tokens of one to three characters, and counting through tokenization is a known weakness (Zhang et al. 2024; Fu et al. 2024). A canvas model can fix the end first and write into the space left. But the canvas is measured in tokens, not characters (here *n* + 6 tokens), so it doesn't pin the length either. Fixed canvas lengths are a known weak point of LLaDA (DAEDAL, Li et al. 2025) and of Dream when infilling (DreamOn, as summarised in the background report).

**Prediction.** Small differences overall. Whatever difference there is should sit where the order of decisions can matter: length, and the final word, if a model applies the tone rule at all.

**Memory is different.** An autoregressive model trained on "A then B" learns p(B | A) and not p(A | B), the reversal curse (Berglund et al. 2023). A masked model trains both directions. The LLaDA paper reports that this lets LLaDA beat GPT-4o at completing the line *before* a given line of a famous poem (Nie et al. 2025). The prediction for the recall set: Qwen's recall should be lopsided toward forward, and LLaDA's symmetric.

## 3. What the data show

### 3.1 Tone: known, but not used

*Do the models know tones?* Probe K asked each model for the Mandarin tone of the 554 line-final characters with one unambiguous reading, in two wordings:
- **K:** "只回答一个数字", with the reply's first position read over the digits 1–4.
- **K2:** the reply started for the model with "第", with the next position read over digits and Chinese numerals together. Qwen put only 1% of its first-token mass on bare digits in K (it wants to answer in words), but 78–86% of every model's mass lands on the K2 candidates, so K2 is the fairer test.

| Model | K: right tone | K: right 平/仄 | K2: right tone | K2: right 平/仄 |
|---|---|---|---|---|
| Qwen | 62% | 88% | 44% | 71% |
| LLaDA | 48% | 66% | 47% | 83% |
| Dream | 42% | 61% | 40% | 67% |

Chance is 25% for the tone and 50% for 平/仄; always answering the commonest tone (4) would score 33%. Every model knows tones well above chance but far from reliably. That is enough to apply the rule much more often than they do.

*Do they use the rule?* Probe S puts the original line minus its last character after the exact phase 1 prompt and reads the last slot's distribution: how much of its mass falls on 平 characters. It does this twice, with the prompt as used (rule 三 states the end tone) and with rule 三 deleted. All 723 items, both directions.

| Model | P(平) writing second lines | P(平) writing first lines | Effect of stating the rule (second / first) |
|---|---|---|---|
| Qwen | 67% | 61% | +1 / +0 points |
| LLaDA | 60% | 51% | +1 / −0 points |
| Dream | 55% | 55% | +0 / −0 points (see below) |

A model that obeyed the rule would move a second line's last character toward 平 and a first line's toward 仄. Deleting the rule changes nothing, and the small direction difference (6–9 points) is the same either way. That difference comes from the 上句/下句 labels, not from the instruction.

**Checks.**
- *Is the slot real?* On well-known couplets, Qwen and LLaDA put most of the slot's mass on the original character (贾雨村风尘怀闺_ → 秀 0.93 and 0.97).
- *Does the end padding matter?* LLaDA gives the same answer with the rest of the canvas masked instead of padded (rule effect +0.1 and −0.5 points).
- *Dream.* Dream's slot can't be read at temperature 1. Only 13% of its mass falls on single 汉字 and 1.5% on the original character; much of the rest goes to stray English subwords (' wearable', ' brokerage'), which also turn up in its phase 1 lines. It produces Chinese only because it samples at 0.2–0.5, which sharpens a weak peak. For Dream, the evidence is its samples (next table), which tell the same story.

*In the actual samples.* The recall set separates following the rule from echoing the given line, because its classical originals often end the "wrong" way. Two effects, in percentage points:
- *Rule effect:* how much more often a line ends on 平 when the rule asks for 平 than when it asks for 仄, with the given line's tone held fixed. +100 is perfect obedience.
- *Imitation effect:* how much more often it ends on 平 when the given line does, with direction held fixed. Positive means echoing the given tone; negative means contrasting with it.

| Writer | Rule effect | Imitation effect | Ends on 平 overall |
|---|---|---|---|
| Classical originals | +2 [−0, +5] | −63 [−68, −57] | 57% |
| Qwen | +4 [+1, +7] | −18 [−21, −14] | 68% |
| LLaDA 0.7 | +7 [+3, +11] | −19 [−22, −16] | 62% |
| Dream 0.5 | −0 [−4, +3] | +39 [+35, +43] | 65% |

(Modern tones. With 平水韵 tones the picture is the same, and sharper for the originals: imitation −90.)

**Reading.** No model follows the rule; the best rule effect, LLaDA at temperature 0, is +10 points out of 100. All of them prefer to end a line on 平 (59–69% of their lines). A plausible source is regulated verse. In a 律诗 the four rhyming lines end on 平 and the first line usually does too, so about five line endings in eight are 平, close to the models' rate. That is an argument from prosody, not something measured here. On 《火候》 the given line always ends the conventional way. A model that ignores the rule but prefers 平 therefore passes when writing second lines and fails when writing first lines, and the stronger its preference, the bigger that gap. On the unseen set Qwen ends 72–78% of its lines on 平 whichever line it writes; LLaDA 56–57%. **That alone produces phase 1's "LLaDA better at first lines, worse at second lines" pattern** (`results.md` §2). It is a difference in bias, not in skill. Which of the two knows tones better depends on how the question is worded (probe K against K2), and neither ordering matches their reverse pass rates. The bias does.

**An exact split of phase 1's end-tone numbers.** On 《火候》 every given first line ends on 仄 and every given second line on 平. So the two pass rates divide, by arithmetic alone, into two terms:
- pass(second lines) − pass(first lines) = 2*b* − 1, where *b* is the model's overall share of 平 endings: **bias**;
- pass(second lines) + pass(first lines) − 1 = P(平 | writing second lines) − P(平 | writing first lines): **how much the ending follows the direction** (rule plus imitation, which the unseen set cannot tell apart).

This is an identity, not a test. What it does is put phase 1's own numbers into the two terms:

| Model | End-tone pass, second / first lines | Bias term (difference) | Direction term (sum − 1) |
|---|---|---|---|
| Qwen | 78% / 28% | +51 | +6 |
| LLaDA 0.7 | 57% / 44% | +13 | +1 |
| Dream 0.5 | 44% / 38% | +5 | −18 |
| LLaDA 0 | 57% / 42% | +15 | −1 |
| Dream 0.2 | 50% / 34% | +15 | −16 |

The direction term is near zero for Qwen and LLaDA, and negative for Dream, which echoes the given line's tone. Phase 1's whole Qwen–LLaDA contrast on end tone sits in the bias term.

**How committed is the last character?** To see what enforcing the rule at the last step would cost, the final slot was read again (`explore/probe_rule_price.py`). This time the line before it was each model's own first sample (bare, right length) instead of the original. Two numbers:
- how often the model's favourite final character already has the class the rule wants;
- when it doesn't, how much less likely the best allowed character is, in nats (1 nat ≈ 2.7 times).

| Model | Set | Second lines: favourite allowed | Median cost if not | First lines: favourite allowed | Median cost if not |
|---|---|---|---|---|---|
| Qwen | unseen | 79% | 2.3 nats | 29% | 4.3 nats |
| Qwen | recall | 73% | 3.3 nats | 36% | 4.4 nats |
| LLaDA 0.7 | unseen | 50% | 1.9 nats | 54% | 1.4 nats |
| LLaDA 0.7 | recall | 63% | 2.4 nats | 41% | 2.8 nats |

By its last character, a model's own line is largely committed. When Qwen's favourite ending for a first line has the wrong tone (seven times in ten on 《火候》), the best 仄 character is typically about 75 times less likely (e^4.3). Forcing the rule only at the last step would therefore often force an awkward character. Meeting it gracefully means choosing the final word earlier. That is the small look-ahead of §2, and it is real. Costs can't be compared across models, because Qwen's distributions are sharper; the "favourite allowed" shares just repeat the 平 bias.

This fits outside evidence. Across 17 LLMs writing 词, keeping the tonal pattern was "substantially harder than structural rules" (Zhao et al. 2026, CCiV). PoeTone treats Songci tone and rhyme as constraints an LLM has to be pushed to meet, by selection or fine-tuning (Qu et al. 2025). PoetryDiffusion's strong meter came from a classifier steering every step, not from diffusion itself (background report).

### 3.2 The classical 回目 keep opposition, not assignment

Classical tones, couplets whose final characters both have an unambiguous class:

| Work | Couplets | First 仄, second 平 (the phase 1 rule) | Final tones differ (opposition) |
|---|---|---|---|
| 水浒传 | 99 | 47% | 92% |
| 红楼梦 | 96 | 46% | 86% |
| 三国演义 | 90 | 42% | 100% |
| 西游记 | 75 | 73% | 100% |
| 镜花缘 (dev) | 75 | 56% | 100% |
| 海上花列传 | 54 | 56% | 98% |
| 儒林外史 | 42 | 60% | 100% |
| 火候 | 28 | 96% | 96% |

The classical 回目 almost always end their two lines on different tones, but which line takes 平 is close to a coin toss. "First line 仄, second line 平" is the 对联 convention. 《火候》 keeps it, the classics keep it about half the time, and that is why the recall set's ceiling in `full_report.md` (table 4) is so low. The models keep opposition at about chance: Qwen and LLaDA 49–60%, Dream 29–42% (it echoes the given tone).

**Consequence for measurement.** On the recall set, the phase 1 end-tone rule measured agreement with a convention the source texts don't follow. Re-scoring phase 1 with opposition in place of assignment (bare condition, both directions, all other rules unchanged; `explore/out/opposition_rescore.md`) shows what that did:

| Writer | Unseen, frozen rule | Unseen, opposition | Recall, frozen rule | Recall, opposition |
|---|---|---|---|---|
| Originals | 88% | 88% | 42% | 77% |
| Qwen | 39% | 37% | 41% | 43% |
| LLaDA 0.7 | 37% | 36% | 35% | 39% |
| LLaDA 0 | 45% | 43% | 41% | 46% |
| Dream 0.5 | 3% | 3% | 6% | 3% |

Under the frozen rule the models looked about as rule-abiding on the recall set as the classical authors themselves: Qwen 41%, the originals 42%. Under the classics' own rule the originals reach 77% and the models stay near 40%, a gap of about 35 points. On 《火候》 the choice of rule makes no difference, because its 回目 keep both.

With opposition, the LLaDA − Qwen differences on the recall set also shrink: −8 instead of −15 writing second lines, and −0 instead of +3 writing first lines. That is what the bias explanation in §3.1 predicts.

### 3.3 Length: LLaDA plans it, Dream doesn't, Qwen counts

The run files record the denoising step at which each canvas position was fixed (`commit_step`), so the order each sampler actually used can be read back:

| Sampler | First token fixed is end padding | All end padding fixed before the line | Within the line, left-to-right correlation | Exact length |
|---|---|---|---|---|
| LLaDA 0.7 | 100% | 84% | +0.17 | 78% |
| LLaDA 0 | 100% | 88% | +0.15 | 83% |
| Dream 0.5 | 13% | 7% | +0.45 | 50% |
| Dream 0.2 | 0% | 0% | +0.70 | 53% |
| Qwen (left to right) | – | – | +1 | 81% |

**LLaDA decides the length first.** Its sampler fixes the most confident position first, and in a canvas a few tokens larger than the line, the most confident positions are the padding at the far end. So in 84–88% of samples LLaDA fills the padding from the right inward until it reaches the space it expects the line to take, and only then writes characters. That is, in effect, the length plan an autoregressive model lacks.

Probe L shows the plan exists before a single token is fixed. On the fully masked canvas:
- LLaDA's most confident position is the canvas's last slot, predicted as an end token, in all 1,446 canvases.
- Its probability of an end token jumps from 0.05 over the line part to 0.94 over the padding.
- The boundary sits on average where the original line ends (−0.09 tokens), and exactly there in 40% of canvases.

**Dream writes almost left to right, and its sampler's arithmetic, not a plan, makes it so.** Read at temperature 1, Dream's beliefs look like LLaDA's: its most certain position is the end padding in every canvas, though it expects the line to run 3.8 tokens longer than the original. But Dream's sampler ranks positions by an entropy computed after three things:
- dividing by the temperature;
- truncating to the top 95% of probability;
- rounding in bf16.

At the phase 1 settings that leaves many positions with entropy exactly zero. On 300 canvases, 7 (temperature 0.5) or 12 (0.2) positions tie for first, mostly in the line, and the sampler tends to take a low one. Its first pick falls in the line in 99–100% of canvases and on the very first position in 47% and 83%. Every step repeats this, so the line fills roughly left to right (correlation +0.45 and +0.70 in the runs) and the padding comes last.

Dream thus gets neither LLaDA's length plan nor an autoregressive model's training to stop. Its lengths are the least accurate, and they depend on the canvas: on the dev set, going from 4 to 6 tokens of slack made its lines 0.9 → 1.7 characters too long on average, while LLaDA's stayed within 0.1.

**The plan buys nothing at this length.** LLaDA's exact-length rate (78%) is no better than Qwen's (81%). At seven or eight characters, counting is easy enough for an autoregressive model. All three are pulled toward the seven-character line of 七言 verse and most 对联:

| Required | Qwen writes 7 | LLaDA 0.7 writes 7 | Dream 0.5 writes 7 |
|---|---|---|---|
| 6 characters (16 item-directions) | 4% | 8% | 54% |
| 8 characters (704 item-directions) | 25% | 18% | 12% |
| 9 characters (8 item-directions) | 20% | 6% | under 2% |

Most items ask for 7 or 8 characters; the 6- and 9-character rows rest on few items.

Nor does the end tone depend much on when LLaDA fixes the line's last token. Writing second lines, the pass rate is 69% when the last token is fixed in the first third of the line's order and 62% when it is fixed in the last third. Writing first lines it is 40% and 45%. The differences are small and point in opposite directions. Being able to choose the end first doesn't help a model that doesn't apply the tone rule.

### 3.4 Parallelism: every model aligns as well as the originals

On answers of the right length, the frozen parallelism score against the line actually given is compared with the same score against another item's given line of the same length and role (a shuffled baseline):

| Writer | Unseen: own − shuffled | Recall: own − shuffled |
|---|---|---|
| Originals | +0.30 [+0.23, +0.37] | +0.21 [+0.19, +0.22] |
| Qwen | +0.26 [+0.23, +0.30] | +0.25 [+0.23, +0.26] |
| LLaDA 0.7 | +0.27 [+0.22, +0.31] | +0.26 [+0.24, +0.28] |
| Dream 0.5 | +0.28 [+0.21, +0.33] | +0.16 [+0.14, +0.19] |

About half the positions match in word class by chance: 回目 share a template (name, verb, object). Above that, Qwen and LLaDA align with the given line as much as the original authors did; Dream does less on the recall set. The parallelism differences in `full_report.md` come almost entirely from wrong-length lines, which score zero. Position-by-position 对仗 is exactly the rule the background report likened to Sudoku's cross-cell constraints. Here it binds a position to known text, so it is not hard for either kind of model.

### 3.5 Relevance is partly copying

With a summary, the share of the answer's characters that also occur in the summary:

| Writer | From the summary | From the given line | Correlation of the summary share with relevance |
|---|---|---|---|
| Originals | 46% | 2% | – |
| Qwen | 48% | 6% | +0.46 |
| LLaDA 0.7 | 74% | 6% | +0.35 |
| Dream 0.5 | 77% | 12% | +0.42 |
| Dream 0.2 | 82% | 9% | +0.28 |

The diffusion models build their lines out of the summary's characters, far more than the original authors or Qwen do. The relevance score (bge-m3 cosine to the summary) rises with that copying. A masked-token objective rewards copying from visible context: any token visible elsewhere in the prompt is a cheap guess for a masked one. Dream also copies the given line, which costs it the no-repeat rule.

**How much of the relevance edge is copying?** Compare Qwen's and the diffusion models' lines that copy the same share of the summary: within five bands of copy share, weighted alike (`explore/out/relevance_vs_copy.md`). Unseen set, relevance minus Qwen's:

| Model | Overall | At the same copy share |
|---|---|---|
| LLaDA 0.7 | +0.061 [+0.045, +0.080] | +0.036 [+0.011, +0.060] |
| LLaDA 0 | +0.063 [+0.043, +0.083] | +0.035 [+0.006, +0.064] |
| Dream 0.5 | +0.022 [+0.006, +0.037] | +0.001 [−0.023, +0.024] |
| Dream 0.2 | +0.042 [+0.025, +0.059] | +0.017 [−0.004, +0.038] |

Copying accounts for about half of LLaDA's relevance edge and all of Dream's. The other half of LLaDA's survives at equal copying. That is a small advantage on the relevance measure that copying doesn't explain, though the author's grades didn't reflect it either.

**The order explains the copying.** An adaptive sampler fixes the easiest positions first (Kim et al. 2025), and for a masked model, copying a token visible in the prompt is the easiest move there is. The commit records show it. Tokens whose characters all come from the summary are fixed well before the line's other tokens (`explore/out/copy_first.md`):
- LLaDA 0.7: mean rank 0.43 against 0.68, where 0 is fixed first and 1 last.
- Within the same third of the line the gap is still −0.22 (LLaDA 0.7), −0.23 (LLaDA 0), −0.19 (Dream 0.5) and −0.09 (Dream 0.2, whose order is mostly set by ties).

So LLaDA anchors each line on characters lifted from the summary and fills the rest around them. A line built that way copies more than one written left to right, where every character must follow from the one before. Copying shapes the order, and the order raises the copying.

The author's grades, if anything, penalise copying. Across the 48 graded lines, the more of a line's characters come from the summary, the lower its grade (Spearman ρ = −0.30, p = 0.04).
- Within Qwen's 16 lines the relation is strong (ρ = −0.64).
- Within LLaDA's 17 and Dream's 15 it is absent (+0.03, −0.02).
- With so few lines this is a hint, not a finding. It points the same way: relevance, as measured, rewards something the author does not.

### 3.6 Memory: the reversal curse, small but clean

Exact recall is rare (`results.md` §3, corrected). The partial-recall score, the share of positions where the answer has the original's character, averaged over all 580 classical items:

| Model | Forward (write the later line) | Reverse (write the earlier line) | Forward − reverse |
|---|---|---|---|
| Qwen | 9.2% | 4.2% | +5.0 [+3.6, +6.5] |
| LLaDA 0.7 | 5.4% | 5.6% | −0.2 [−1.1, +0.7] |
| Dream 0.5 | 1.2% | 1.1% | +0.1 [−0.3, +0.4] |

LLaDA − Qwen: −3.8 [−5.2, −2.5] forward, +1.4 [+0.6, +2.2] reverse. This is the theory's prediction and the LLaDA paper's poem result, on 回目:
- the autoregressive model stores the couplet one way;
- the masked model stores it less strongly but both ways, and beats the autoregressive model on the way it doesn't store.

In exact terms: Qwen reproduced 30 second lines and 4 first lines; LLaDA 1 and 4.

**Dream is the odd one out.** It starts from Qwen2.5's weights, which recall 30 couplets, yet it recalls almost none. Converting the model to diffusion seems to have overwritten the verbatim memory rather than turned it into two-way memory. This study cannot tell whether that comes from the continued training or from Dream's broken output format.

## 4. What this means for the phase 1 verdict

The verdict stands: neither diffusion model wrote better 回目 than Qwen. The theory adds four things:
- **It is the expected result.** With one line given, only length and the final word need any look-ahead, and nobody applied the tone rule, so there was little for a different generation order to win.
- **The one "pattern that holds" is a bias effect.** The direction asymmetry comes from how strongly each model prefers 平 endings, and none of them reacts to the rule. It should not be cited as diffusion being better at first lines.
- **The real differences are in the machinery, not the quality.** LLaDA settles length first; Dream's sampler writes left to right; LLaDA's memory is symmetric. None of these turns into better couplets at seven or eight characters.
- **The recall set measured something other than intended.** The classics keep tone opposition, not the tested assignment, and the models had memorized little. The unseen−recall contrast is "new against familiar", with a tone rule that only one side of it follows.

## 4b. Order or training? What phase 1 says about the phase 3 question

The background report frames phase 3 as two competing hypotheses: a behaviour comes from the generation order, or it lives in the weights. Phase 1 is too small a task to settle that, but it gives a warning about how to test it.

| Difference seen in phase 1 | Comes from |
|---|---|
| LLaDA settles the length first | the sampler's confidence rule, together with weights trained to pad answers with end tokens (LLaDA's fine-tuning fills unused positions with end-of-text) |
| Dream writes almost left to right | the **sampler's numerics**: at low temperature, truncation and bf16 rounding tie many positions at zero entropy, and ties go to the leftmost. Dream's own beliefs would fix the padding first |
| Symmetric memory in LLaDA, forward-only in Qwen, almost none in Dream | the training objective (and, for Dream, the conversion) |
| A preference for 平 endings, no use of the stated rule | the weights: pretraining habits and instruction tuning, in all three |
| Heavy copying from the summary in both diffusion models | the masked-token objective, in both |

**The warning.** An adaptive sampler's order is not chosen independently of everything else. It *emerges* from the model's confidence, which training shaped, and from the sampler's own arithmetic: temperature, truncation, numeric precision, tie-breaking. Dream at its default settings is a diffusion model whose effective order is close to left to right (correlation +0.70). That comes not from what the model believes but from how its sampler breaks ties. Comparing "Dream's diffusion mode" with "Qwen's autoregressive mode" would compare two nearly left-to-right generators, and could find no order effect simply because the orders hardly differed.

So a phase 3 test of order needs three things:
- **force** the orders being compared (left-to-right, right-to-left, anchor-first), rather than trusting the sampler's default;
- **log the commit order,** as phase 1 did, to show the orders really differed;
- prefer a model whose default order is not already left to right. LLaDA, trained from scratch, qualifies; a block model like Nemotron-Diffusion or LLaDA2.x should be checked first.

## 5. Predictions for later phases (not tested)

These follow from the theory and the probes, so they can be tested later.

1. **Fixed later anchors (phase 2).** When a later fixed point constrains earlier text (a payoff, a pinned final character, a clue that must be planted before its use), the autoregressive model meets the rule's lookahead case, and the canvas model can condition on the anchor directly. Here the theory predicts a diffusion advantage that phase 1 could not show. Measure:
   - whether the anchor is reached at all;
   - the fluency of the text just before it;
   - how often the planted idea is stated outright.
2. **Order on the same weights (phase 3).** Forcing LLaDA to fill left to right should remove its length-first behaviour. If length-first is what keeps LLaDA's length accurate, its exact-length rate should fall below Qwen's. Its tone and parallelism should hardly change, because they never depended on order. Right-to-left order fixes the final character first and, by §3.3, should not raise the end-tone rate.
3. **Rules applied by the decoder: the one place where order should matter even with one line given.** The models ignore the stated rule and know tones only moderately, so the rule is better enforced from outside, by a decoder that allows only correct-class characters in the final slot (judged by a tone table like the checker's). Any model then keeps the end tone. Where it is enforced matters (§3.1):
   - *at the last step,* the only option for an autoregressive model, it lands on a line already committed to the wrong ending and forces a character often tens of times less likely;
   - *from the first step,* which a canvas model allows, the restricted slot is in place while the rest of the line is still open, so the line can grow toward an allowed ending.

   The prediction, testable on LLaDA alone with the same weights: restricting the final slot from the start gives rule-keeping lines that the model itself rates as more likely, and the author grades higher, than restricting it only at the end. As PoetryDiffusion showed, the form comes from guidance; the canvas decides how cheaply the guidance can be met.
4. **A canvas counted in characters.** If every canvas position held exactly one character, diffusion could fix the length by construction. The token canvas used here couldn't, which removed the one advantage phase 1 could have shown.

## 6. If phase 1 were run again: measurement lessons

- **Score tone by direction contrast, not pass rate.** Report the rule effect, and opposition alongside assignment. Include given lines whose final tone varies, or bias will pass for skill.
- **Report parallelism above a shuffled baseline,** on right-length lines only, and count length failures separately.
- **Control relevance for copying,** for example by comparing models at the same share of copied characters, as in §3.5, where it halves LLaDA's edge and removes Dream's.
- **Count memorization per item and per direction,** and add partial recall. Per-attempt rates rounded to whole percent hid the reversal-curse pattern in `results.md`.

## 7. Limits of this note

- **Post hoc:** every analysis was chosen after the results were seen; there is no correction for multiplicity.
- **Probes read distributions, not samples.** They are single forward passes, teacher-forced in probe S. They show what the model's distribution favours, not what the sampler then produces.
- **Tones:** modern tones come from pypinyin and classical tones from a 平水韵 table. Characters with more than one reading, or listed in both classes, were dropped. The tone probe is in the models' own words ("第几声"), so a wrong answer may partly mean a misunderstood question.
- **Commit order describes the sampler,** not any intention in the model.
- **Sampling precision is not a confound here.** Low-precision categorical sampling quietly lowers a masked model's effective temperature (Zheng et al. 2024). The LLaDA sampler drew its Gumbel noise in float64, as in the reference code, so that effect does not apply.
- **One run per arm, one tagger, one embedding model; 43 unseen items.**

## References

Checked through the arXiv API on 2026-11-26.

- Bachmann, G. & Nagarajan, V. (2024). The pitfalls of next-token prediction. arXiv:2403.06963.
- Berglund, L. et al. (2023). The Reversal Curse: LLMs trained on "A is B" fail to learn "B is A". arXiv:2309.12288.
- Fu, T. et al. (2024). Why Do Large Language Models (LLMs) Struggle to Count Letters? arXiv:2412.18626.
- Hoogeboom, E. et al. (2021). Autoregressive Diffusion Models. arXiv:2110.02037.
- Kim, J. et al. (2025). Train for the Worst, Plan for the Best: Understanding Token Ordering in Masked Diffusions. arXiv:2502.06768.
- Li, J. et al. (2025). Beyond Fixed: Training-Free Variable-Length Denoising for Diffusion Large Language Models (DAEDAL). arXiv:2508.00819.
- Nie, S. et al. (2025). Large Language Diffusion Models (LLaDA). arXiv:2502.09992.
- Ou, J. et al. (2024). Your Absorbing Discrete Diffusion Secretly Models the Conditional Distributions of Clean Data. arXiv:2406.03736.
- Peng, F. Z. et al. (2025). Path Planning for Masked Diffusion Model Sampling. arXiv:2502.03540.
- Qu, Z. et al. (2025). PoeTone: A Framework for Constrained Generation of Structured Chinese Songci with LLMs. arXiv:2508.02515.
- Shih, A. et al. (2022). Training and Inference on Any-Order Autoregressive Models the Right Way. arXiv:2205.13554.
- Ye, J. et al. (2024). Beyond Autoregression: Discrete Diffusion for Complex Reasoning and Planning. arXiv:2410.14157.
- Ye, J. et al. (2025). Dream 7B: Diffusion Large Language Models. arXiv:2508.15487.
- Zhang, X. et al. (2024). Counting Ability of Large Language Models and Impact of Tokenization. arXiv:2410.19730.
- Zhao, S. et al. (2026). CCiV: A Benchmark for Structure, Rhythm and Quality in LLM-Generated Chinese Ci Poetry. arXiv:2602.14081.
- Zheng, K. et al. (2024). Masked Diffusion Models are Secretly Time-Agnostic Masked Models and Exploit Inaccurate Categorical Sampling. arXiv:2409.02908.

DreamOn and PoetryDiffusion are cited as reported in the author's background report (`../docs/background-research.md`), not re-read here.
