## 1. Pass rates and scores

Per sample. “All rules” = length, modern end tone and no repeat together, and not a copy of the given line; its bracket is a 95% bootstrap interval over items. Parallelism counts a wrong-length line as 0. Relevance is cosine similarity to the summary (bge-m3).

| Set | Condition | Direction | Model | Items | Length | End tone (modern) | End tone (classical) | No repeat | All rules | All rules (classical tone) | Parallelism | Relevance | Exact match | Distinct per 8 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| dev | summary | forward | qwen | 8 | 84% | 84% | 88% | 89% | 66% [42, 88] | 67% | 0.69 | 0.48 | 0% | 6.4 |
| dev | summary | forward | dream | 8 | 3% | 62% | 52% | 80% | 0% [0, 0] | 0% | 0.04 | 0.54 | 0% | 5.5 |
| dev | summary | forward | llada | 8 | 86% | 47% | 50% | 95% | 34% [17, 53] | 38% | 0.74 | 0.55 | 0% | 6.4 |
| dev | summary | forward | dream-default | 8 | 0% | 39% | 39% | 100% | 0% [0, 0] | 0% | 0.00 | 0.57 | 0% | 2.1 |
| dev | summary | forward | llada-default | 8 | 100% | 38% | 50% | 100% | 38% [12, 75] | 50% | 0.88 | 0.56 | 0% | 1.0 |
| dev | summary | reverse | qwen | 8 | 53% | 34% | 36% | 66% | 19% [3, 39] | 20% | 0.40 | 0.50 | 0% | 6.0 |
| dev | summary | reverse | dream | 8 | 3% | 31% | 44% | 62% | 0% [0, 0] | 0% | 0.06 | 0.60 | 0% | 6.1 |
| dev | summary | reverse | llada | 8 | 80% | 42% | 59% | 83% | 28% [9, 50] | 42% | 0.65 | 0.56 | 0% | 6.1 |
| dev | summary | reverse | dream-default | 8 | 12% | 31% | 50% | 50% | 0% [0, 0] | 0% | 0.11 | 0.59 | 0% | 1.5 |
| dev | summary | reverse | llada-default | 8 | 75% | 50% | 75% | 88% | 12% [0, 38] | 38% | 0.67 | 0.56 | 0% | 1.0 |
| dev | bare | forward | qwen | 10 | 100% | 74% | 74% | 92% | 68% [46, 88] | 69% | 0.85 | 0.38 | 0% | 6.2 |
| dev | bare | forward | dream | 10 | 20% | 69% | 62% | 56% | 10% [0, 28] | 10% | 0.40 | 0.41 | 0% | 6.4 |
| dev | bare | forward | llada | 10 | 86% | 62% | 60% | 78% | 42% [28, 56] | 39% | 0.71 | 0.42 | 0% | 6.9 |
| dev | bare | forward | dream-default | 10 | 10% | 69% | 69% | 71% | 0% [0, 0] | 0% | 0.45 | 0.41 | 0% | 1.3 |
| dev | bare | forward | llada-default | 10 | 100% | 50% | 50% | 90% | 40% [10, 70] | 40% | 0.82 | 0.46 | 0% | 1.0 |
| dev | bare | reverse | qwen | 10 | 76% | 31% | 46% | 92% | 19% [6, 39] | 34% | 0.68 | 0.41 | 0% | 6.0 |
| dev | bare | reverse | dream | 10 | 38% | 34% | 36% | 78% | 10% [0, 25] | 10% | 0.62 | 0.47 | 0% | 6.5 |
| dev | bare | reverse | llada | 10 | 86% | 32% | 36% | 76% | 26% [10, 44] | 29% | 0.76 | 0.43 | 0% | 7.5 |
| dev | bare | reverse | dream-default | 10 | 40% | 39% | 39% | 69% | 10% [0, 30] | 10% | 0.68 | 0.46 | 0% | 1.7 |
| dev | bare | reverse | llada-default | 10 | 90% | 20% | 30% | 70% | 10% [0, 30] | 10% | 0.81 | 0.43 | 0% | 1.0 |

## 3. Diffusion model minus Qwen, on the same items

Percentage points (score points for parallelism and relevance), 95% interval, * = interval excludes zero.

| Set | Condition | Direction | Model | All rules | Length | End tone | No repeat | Parallelism | Relevance | Exact match |
|---|---|---|---|---|---|---|---|---|---|---|
| dev | summary | forward | dream − qwen | -66 [-88, -42] * | -81 [-97, -62] * | -22 [-55, +9] | -9 [-38, +11] | -0.65 [-0.80, -0.48] * | +0.05 [+0.00, +0.10] * | +0 [+0, +0] |
| dev | summary | forward | llada − qwen | -31 [-62, +5] | +2 [-19, +27] | -38 [-62, -12] * | +6 [-5, +17] | +0.04 [-0.11, +0.23] | +0.07 [+0.03, +0.11] * | +0 [+0, +0] |
| dev | summary | forward | dream-default − qwen | -66 [-88, -42] * | -84 [-98, -66] * | -45 [-80, -8] * | +11 [+3, +22] * | -0.69 [-0.85, -0.52] * | +0.09 [+0.05, +0.13] * | +0 [+0, +0] |
| dev | summary | forward | llada-default − qwen | -28 [-70, +17] | +16 [+2, +34] * | -47 [-83, -6] * | +11 [+3, +22] * | +0.19 [+0.00, +0.38] * | +0.08 [+0.03, +0.13] * | +0 [+0, +0] |
| dev | summary | reverse | dream − qwen | -19 [-39, -3] * | -50 [-77, -23] * | -3 [-23, +20] | -3 [-52, +45] | -0.34 [-0.60, -0.06] * | +0.09 [+0.06, +0.13] * | +0 [+0, +0] |
| dev | summary | reverse | llada − qwen | +9 [-16, +38] | +27 [+6, +52] * | +8 [-20, +41] | +17 [-23, +55] | +0.25 [+0.08, +0.45] * | +0.06 [+0.03, +0.08] * | +0 [+0, +0] |
| dev | summary | reverse | dream-default − qwen | -19 [-39, -3] * | -41 [-70, -14] * | -3 [-25, +19] | -16 [-56, +27] | -0.29 [-0.49, -0.10] * | +0.09 [+0.05, +0.15] * | +0 [+0, +0] |
| dev | summary | reverse | llada-default − qwen | -6 [-36, +28] | +22 [-8, +56] | +16 [-25, +55] | +22 [-22, +61] | +0.27 [-0.02, +0.55] | +0.06 [+0.01, +0.10] * | +0 [+0, +0] |
| dev | bare | forward | dream − qwen | -58 [-78, -36] * | -80 [-100, -50] * | -5 [-39, +26] | -36 [-61, -14] * | -0.45 [-0.63, -0.25] * | +0.03 [-0.02, +0.09] | +0 [+0, +0] |
| dev | bare | forward | llada − qwen | -25 [-46, +0] | -14 [-21, -7] * | -11 [-30, +8] | -15 [-35, +4] | -0.14 [-0.27, -0.02] * | +0.05 [+0.01, +0.10] * | +0 [+0, +0] |
| dev | bare | forward | dream-default − qwen | -68 [-88, -46] * | -90 [-100, -70] * | -5 [-44, +30] | -21 [-54, +10] | -0.40 [-0.65, -0.15] * | +0.04 [-0.01, +0.08] | +0 [+0, +0] |
| dev | bare | forward | llada-default − qwen | -28 [-66, +14] | +0 [+0, +0] | -24 [-60, +11] | -3 [-26, +19] | -0.03 [-0.14, +0.06] | +0.08 [+0.03, +0.13] * | +0 [+0, +0] |
| dev | bare | reverse | dream − qwen | -9 [-34, +14] | -39 [-65, -14] * | +3 [-35, +36] | -15 [-31, -3] * | -0.06 [-0.25, +0.15] | +0.06 [+0.01, +0.10] * | +0 [+0, +0] |
| dev | bare | reverse | llada − qwen | +8 [-20, +32] | +10 [-6, +29] | +1 [-34, +35] | -16 [-38, +3] | +0.08 [-0.08, +0.27] | +0.02 [-0.01, +0.05] | +0 [+0, +0] |
| dev | bare | reverse | dream-default − qwen | -9 [-36, +19] | -36 [-66, -7] * | +8 [-35, +48] | -24 [-52, +2] | +0.01 [-0.33, +0.33] | +0.05 [+0.02, +0.10] * | +0 [+0, +0] |
| dev | bare | reverse | llada-default − qwen | -9 [-36, +20] | +14 [-18, +43] | -11 [-48, +28] | -23 [-54, +6] | +0.13 [-0.19, +0.43] | +0.02 [-0.02, +0.06] | +0 [+0, +0] |

## 4. The original couplets under the same checker

Each original line scored as if it had been generated from the other. This is the ceiling for the rules.

| Set | Direction | Items | Length | End tone (modern) | End tone (classical) | No repeat | All rules | All rules (classical tone) | Parallelism |
|---|---|---|---|---|---|---|---|---|---|
| dev | forward | 10 | 100% | 60% | 60% | 100% | 60% | 60% | 0.87 |
| dev | reverse | 10 | 100% | 60% | 60% | 100% | 60% | 60% | 0.87 |
