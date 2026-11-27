# Exact recall by item and direction

Post hoc; see `PROTOCOL.md` entry 37. An item counts as recalled in a direction if any attempt, with or without the summary, reproduced the original line exactly. 580 classical items. *Forward*: given the first line, the original second line. *Reverse*: given the second line, the original first line. Fisher's exact test compares the two directions' item counts.

| Model | Forward: items (attempts) | Reverse: items (attempts) | Forward / reverse | Fisher p |
|---|---|---|---|---|
| Qwen | 30 (147 of 5024) | 4 (7 of 5024) | 7.5 | 4.6e-06 |
| LLaDA 0.7 | 1 (3 of 5024) | 4 (15 of 5024) | 0.2 | 0.37 |
| LLaDA 0 | 1 (1 of 628) | 3 (3 of 628) | 0.3 | 0.62 |
| Dream 0.5 | 0 (0 of 5024) | 0 (0 of 5024) | – | 1 |
| Dream 0.2 | 0 (0 of 5024) | 0 (0 of 5024) | – | 1 |

Direction profile, Qwen against LLaDA 0.7, among recalled items (30:4 against 1:4): Fisher p = 0.0039.

Recalled items by work (Qwen, forward): 水浒传 10, 红楼梦 9, 西游记 7, 三国演义 4.

## Partial recall

Share of positions where the answer has the original's character, averaged per item, over all 580 items (no selection), both conditions. 95% bootstrap intervals over items. A model that stores a couplet in one direction only should score higher forward than reverse; one that stores it symmetrically should score about the same.

| Model | Forward | Reverse | Forward − reverse |
|---|---|---|---|
| Qwen | 9.2% | 4.2% | +5.0 [+3.6, +6.5] |
| LLaDA 0.7 | 5.4% | 5.6% | -0.2 [-1.1, +0.7] |
| LLaDA 0 | 5.8% | 6.3% | -0.5 [-1.8, +0.8] |
| Dream 0.5 | 1.2% | 1.1% | +0.1 [-0.3, +0.4] |
| Dream 0.2 | 1.4% | 1.5% | -0.1 [-0.6, +0.4] |

LLaDA 0.7 − Qwen, forward: -3.8 points [-5.2, -2.5].

LLaDA 0.7 − Qwen, reverse: +1.4 points [+0.6, +2.2].

