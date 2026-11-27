# Tone probes

Post hoc; see `PROTOCOL.md` entry 36 and `explore/probe_tones.py`. Read from the models' output distributions in one forward pass; nothing is sampled.

## Probe K: does the model know a character's Mandarin tone?

Question: 汉字“X”在普通话里读第几声？只回答一个数字。 The first position of the reply, restricted to the tokens 1–4. Characters: the final characters of every original line, with one unambiguous reading. *Tone*: the most probable digit is the right tone (chance 25%; always answering the most common tone scores the column *Majority*). *平/仄*: the mass on 1+2 against 3+4 picks the right class (chance 50%). *On digits*: share of the whole distribution that falls on the four digits.

| Model | Characters | Tone | Majority | 平/仄 | Mean P(right tone) | On digits |
|---|---|---|---|---|---|---|
| Qwen | 554 | 62% | 33% (tone 4) | 88% | 0.61 | 1% |
| LLaDA | 554 | 48% | 33% (tone 4) | 66% | 0.35 | 47% |
| Dream | 554 | 42% | 33% (tone 4) | 61% | 0.33 | 9% |

### K2: the same question, reply started with 第 (`explore/probe_checks.py`)

The next position is read over the digits and the Chinese numerals 一二三四 (and 一声 where it is one token), summed per tone. *On numerals*: share of the whole distribution on those candidates.

| Model | Tone | 平/仄 | On numerals |
|---|---|---|---|
| Qwen | 44% | 71% | 78% |
| LLaDA | 47% | 83% | 86% |
| Dream | 40% | 67% | 84% |

## Probe S: the last character's tone, everything else in place

The exact phase 1 prompt (bare), then the original line minus its last character; the slot that follows is read as a distribution over tokens that are one unambiguous 汉字, and the share of that mass on 平 is recorded. *Rule*: the prompt as used, whose rule 三 states the end tone. *No rule*: the same prompt with rule 三 removed. Forward = writing a second line (the rule wants 平), reverse = a first line (the rule wants 仄). All 723 items of the unseen, recall and dev sets. Intervals: 95% bootstrap over items.

| Model | Prompt | P(平), writing second lines | P(平), writing first lines | Rule effect at the slot | P(original character) |
|---|---|---|---|---|---|
| Qwen | rule | 67% | 61% | +6 [+1, +12] | 0.50 |
| Qwen | no rule | 66% | 61% | +6 [+0, +11] | 0.51 |
| LLaDA | rule | 60% | 51% | +9 [+4, +14] | 0.33 |
| LLaDA | no rule | 59% | 51% | +8 [+3, +13] | 0.34 |
| Dream | rule | 55% | 55% | -0 [-2, +2] | 0.02 |
| Dream | no rule | 54% | 55% | -0 [-2, +1] | 0.02 |

### The same with the rest of the canvas masked (`explore/probe_S_masked_tail.py`)

| Model | Prompt | P(平), second lines | P(平), first lines | P(original character) | Mass on single 汉字 |
|---|---|---|---|---|---|
| LLaDA | rule | 57% | 49% | 0.321 | 67% |
| LLaDA | no rule | 57% | 49% | 0.323 | 67% |
| Dream | rule | 54% | 55% | 0.015 | 13% |
| Dream | no rule | 54% | 55% | 0.014 | 12% |

### What stating the rule does to the slot

P(平) with the rule minus P(平) without it, paired by item. A model that obeys the rule should move toward 平 when writing second lines (positive) and toward 仄 when writing first lines (negative).

| Model | Writing second lines | Writing first lines |
|---|---|---|
| Qwen | +1 [+1, +1] | +0 [-0, +1] |
| LLaDA | +1 [+0, +1] | -0 [-1, +0] |
| Dream | +0 [+0, +0] | -0 [-0, +0] |

### Rule or imitation at the slot (recall set, rule prompt)

As in `existing_data.md`, but for the slot distribution: P(平) by direction and by the given line's final tone (modern).

| Model | Fwd, given 仄 | Fwd, given 平 | Rev, given 仄 | Rev, given 平 | Rule effect | Imitation effect |
|---|---|---|---|---|---|---|
| Qwen | 91% | 44% | 88% | 43% | +2 | -46 |
| LLaDA | 81% | 41% | 80% | 33% | +5 | -44 |
| Dream | 53% | 57% | 53% | 56% | +0 | +3 |

