# Phase 1 re-scored with tone opposition

Post hoc; see `PROTOCOL.md` entry 36. *Frozen*: the phase 1 all-rules rate (length, assigned end tone, no repeat, not a copy). *Opposition*: the same, with the assigned end tone replaced by "the two lines end on different tones" (modern tones; a line whose final tone is unknown fails). Bare condition, both directions pooled; 95% bootstrap intervals over items.

| Writer | Unseen, frozen | Unseen, opposition | Recall, frozen | Recall, opposition |
|---|---|---|---|---|
| Originals (both lines) | 88% [79, 98] | 88% [79, 98] | 42% [38, 46] | 77% [74, 81] |
| Qwen | 39% [33, 45] | 37% [30, 43] | 41% [39, 43] | 43% [41, 46] |
| LLaDA 0.7 | 37% [32, 42] | 36% [31, 41] | 35% [33, 37] | 39% [38, 41] |
| Dream 0.5 | 3% [1, 5] | 3% [1, 5] | 6% [5, 8] | 3% [2, 4] |
| LLaDA 0 | 45% [35, 56] | 43% [33, 53] | 41% [38, 44] | 46% [43, 49] |
| Dream 0.2 | 3% [0, 8] | 2% [0, 6] | 5% [3, 6] | 3% [2, 4] |

The originals' row is a ceiling for the rules only, with each couplet counted once; the frozen ceiling in `full_report.md` table 4 is computed per direction, so it differs slightly.

### LLaDA 0.7 − Qwen under each rule (bare, paired by item)

| Set | Direction | Frozen | Opposition |
|---|---|---|---|
| unseen | forward | -18 [-30, -5] * | -15 [-27, -1] * |
| unseen | reverse | +13 [+4, +21] * | +13 [+4, +21] * |
| recall | forward | -15 [-19, -12] * | -8 [-11, -5] * |
| recall | reverse | +3 [+0, +6] * | -0 [-3, +3] |

### The author's grades against copying from the summary

48 graded lines. Spearman correlation between the share of the line's 汉字 found in the summary and the grade: ρ = -0.30 (p = 0.04).

| Model | Lines | Mean share from the summary | Mean grade |
|---|---|---|---|
| Qwen | 16 | 56% | 2.88 |
| LLaDA 0.7 | 17 | 79% | 2.71 |
| Dream 0.5 | 15 | 74% | 1.60 |

Within each model: Qwen ρ = -0.64 (n = 16); LLaDA 0.7 ρ = +0.03 (n = 17); Dream 0.5 ρ = -0.02 (n = 15).

