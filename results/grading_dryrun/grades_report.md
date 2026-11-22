# Human grades, unblinded

9 of 9 lines graded.

Literary quality, 1 to 5, by the author. One line per model for each of the same chapters and directions. Brackets are 95% bootstrap intervals over those chapter-direction cells.

| Model | Lines | Mean score | Lines scored 4 or 5 | Drawn from rule-passing samples |
|---|---|---|---|---|
| qwen | 3 | 4.00 [4.00, 4.00] | 3 | 3 |
| dream | 3 | 1.33 [1.00, 2.00] | 0 | 0 |
| llada | 3 | 3.00 [2.00, 4.00] | 1 | 0 |

## Differences on the same cells

| Comparison | Cells | Difference in mean score |
|---|---|---|
| dream − qwen | 3 | -2.67 [-3.00, -2.00] * |
| llada − qwen | 3 | -1.00 [-2.00, +0.00] |
| llada − dream | 3 | +1.67 [+1.00, +3.00] * |

## By direction

| Model | Forward (writes the second line) | Reverse (writes the first line) |
|---|---|---|
| qwen | 4.00 [4.00, 4.00] | 4.00 [4.00, 4.00] |
| dream | 1.50 [1.00, 2.00] | 1.00 [1.00, 1.00] |
| llada | 3.50 [3.00, 4.00] | 2.00 [2.00, 2.00] |

## Every graded line

| Entry | Chapter | Direction | Model | Line | Passed all rules | Score | Comment |
|---|---|---|---|---|---|---|---|
| e07 | jhy-003 | forward | dream | 驼宾王写信托子 | no | 2 |  |
| e05 | jhy-003 | forward | llada | 骆宾王寄书托子友 | no | 3 |  |
| e01 | jhy-003 | forward | qwen | 骆临海遗书托后人 | yes | 4 |  |
| e08 | jhy-007 | reverse | dream | 唐小山夜中问叔，是否开女 | no | 1 |  |
| e03 | jhy-007 | reverse | llada | 老书生海外访名花 | no | 2 |  |
| e02 | jhy-007 | reverse | qwen | 女公子夜半话科场 | yes | 4 |  |
| e04 | jhy-033 | forward | dream | 宫中流出，被板子打 | no | 1 |  |
| e06 | jhy-033 | forward | llada | 红颜女穿耳遭痛 | no | 4 |  |
| e09 | jhy-033 | forward | qwen | 青丝客蓄髯遭疼 | yes | 4 |  |
