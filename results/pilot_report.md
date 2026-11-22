## 1. Pass rates and scores

Per sample. “All rules” = length, modern end tone and no repeat together, and not a copy of the given line; its bracket is a 95% bootstrap interval over items. Parallelism counts a wrong-length line as 0. Relevance is cosine similarity to the summary (bge-m3).

| Set | Condition | Direction | Model | Items | Length | End tone (modern) | End tone (classical) | No repeat | All rules | All rules (classical tone) | Parallelism | Relevance | Exact match | Distinct per 8 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| dev | summary | forward | qwen | 8 | 100% | 84% | 84% | 92% | 77% [56, 92] | 77% | 0.86 | 0.50 | 0% | 4.9 |
| dev | summary | forward | dream | 8 | 12% | 47% | 41% | 80% | 5% [0, 14] | 0% | 0.06 | 0.48 | 0% | 5.1 |
| dev | summary | forward | llada | 8 | 84% | 44% | 42% | 98% | 34% [14, 58] | 33% | 0.70 | 0.55 | 0% | 6.6 |
| dev | summary | forward | dream-default | 8 | 38% | 25% | 38% | 75% | 0% [0, 0] | 12% | 0.29 | 0.54 | 0% | 1.4 |
| dev | summary | forward | llada-default | 8 | 100% | 50% | 25% | 100% | 50% [12, 88] | 25% | 0.81 | 0.55 | 0% | 1.0 |
| dev | summary | reverse | qwen | 8 | 70% | 33% | 38% | 88% | 27% [9, 47] | 31% | 0.58 | 0.51 | 0% | 5.8 |
| dev | summary | reverse | dream | 8 | 9% | 53% | 62% | 69% | 9% [0, 28] | 9% | 0.09 | 0.53 | 0% | 5.6 |
| dev | summary | reverse | llada | 8 | 88% | 47% | 64% | 92% | 42% [14, 69] | 58% | 0.71 | 0.60 | 0% | 6.1 |
| dev | summary | reverse | dream-default | 8 | 25% | 56% | 64% | 88% | 25% [0, 62] | 25% | 0.23 | 0.56 | 0% | 1.2 |
| dev | summary | reverse | llada-default | 8 | 88% | 38% | 62% | 75% | 38% [12, 75] | 50% | 0.79 | 0.60 | 0% | 1.0 |
| dev | bare | forward | qwen | 10 | 94% | 79% | 76% | 94% | 72% [52, 90] | 71% | 0.82 | 0.37 | 0% | 6.0 |
| dev | bare | forward | dream | 10 | 18% | 61% | 70% | 78% | 14% [1, 35] | 15% | 0.34 | 0.40 | 0% | 6.4 |
| dev | bare | forward | llada | 10 | 85% | 84% | 84% | 78% | 54% [38, 70] | 54% | 0.67 | 0.41 | 0% | 7.9 |
| dev | bare | forward | dream-default | 10 | 0% | 57% | 57% | 86% | 0% [0, 0] | 0% | 0.32 | 0.41 | 0% | 2.1 |
| dev | bare | forward | llada-default | 10 | 90% | 90% | 90% | 80% | 70% [40, 100] | 70% | 0.78 | 0.44 | 0% | 1.0 |
| dev | bare | reverse | qwen | 10 | 76% | 35% | 48% | 96% | 28% [8, 52] | 36% | 0.66 | 0.41 | 0% | 6.5 |
| dev | bare | reverse | dream | 10 | 22% | 38% | 41% | 74% | 4% [0, 11] | 4% | 0.42 | 0.44 | 0% | 7.1 |
| dev | bare | reverse | llada | 10 | 85% | 29% | 36% | 79% | 22% [12, 34] | 25% | 0.72 | 0.40 | 0% | 7.8 |
| dev | bare | reverse | dream-default | 10 | 16% | 31% | 31% | 69% | 0% [0, 0] | 0% | 0.56 | 0.44 | 0% | 2.6 |
| dev | bare | reverse | llada-default | 10 | 90% | 20% | 30% | 90% | 20% [0, 50] | 20% | 0.78 | 0.43 | 0% | 1.0 |

## 3. Diffusion model minus Qwen, on the same items

Percentage points (score points for parallelism and relevance), 95% interval, * = interval excludes zero.

| Set | Condition | Direction | Model | All rules | Length | End tone | No repeat | Parallelism | Relevance | Exact match |
|---|---|---|---|---|---|---|---|---|---|---|
| dev | summary | forward | dream − qwen | -72 [-92, -44] * | -88 [-100, -70] * | -38 [-80, +6] | -12 [-25, -3] * | -0.80 [-0.89, -0.70] * | -0.02 [-0.08, +0.05] | +0 [+0, +0] |
| dev | summary | forward | llada − qwen | -42 [-67, -19] * | -16 [-27, -6] * | -41 [-66, -19] * | +6 [-3, +20] | -0.16 [-0.29, -0.06] * | +0.05 [-0.00, +0.10] | +0 [+0, +0] |
| dev | summary | forward | dream-default − qwen | -77 [-92, -56] * | -62 [-100, -25] * | -59 [-88, -30] * | -17 [-41, +0] | -0.57 [-0.83, -0.28] * | +0.04 [-0.02, +0.09] | +0 [+0, +0] |
| dev | summary | forward | llada-default − qwen | -27 [-61, +6] | +0 [+0, +0] | -34 [-64, -6] * | +8 [+0, +20] | -0.06 [-0.19, +0.09] | +0.06 [+0.01, +0.10] * | +0 [+0, +0] |
| dev | summary | reverse | dream − qwen | -17 [-44, +11] | -61 [-84, -34] * | +20 [-11, +53] | -19 [-55, +19] | -0.49 [-0.68, -0.28] * | +0.02 [-0.01, +0.05] | +0 [+0, +0] |
| dev | summary | reverse | llada − qwen | +16 [-17, +47] | +17 [+6, +30] * | +14 [-22, +47] | +5 [-14, +28] | +0.13 [+0.04, +0.24] * | +0.08 [+0.05, +0.12] * | +0 [+0, +0] |
| dev | summary | reverse | dream-default − qwen | -2 [-28, +30] | -45 [-78, -11] * | +23 [-6, +52] | +0 [+0, +0] | -0.35 [-0.62, -0.05] * | +0.05 [+0.01, +0.08] * | +0 [+0, +0] |
| dev | summary | reverse | llada-default − qwen | +11 [-23, +47] | +17 [+6, +31] * | +5 [-30, +42] | -12 [-38, +0] | +0.21 [+0.10, +0.33] * | +0.08 [+0.02, +0.15] * | +0 [+0, +0] |
| dev | bare | forward | dream − qwen | -59 [-80, -36] * | -76 [-94, -51] * | -17 [-45, +9] | -16 [-34, -1] * | -0.48 [-0.69, -0.29] * | +0.04 [-0.03, +0.10] | +0 [+0, +0] |
| dev | bare | forward | llada − qwen | -19 [-44, +3] | -9 [-21, +5] | +5 [-11, +24] | -16 [-36, +0] | -0.15 [-0.29, -0.01] * | +0.05 [+0.01, +0.08] * | +0 [+0, +0] |
| dev | bare | forward | dream-default − qwen | -72 [-90, -52] * | -94 [-100, -84] * | -21 [-52, +12] | -7 [-34, +14] | -0.50 [-0.71, -0.27] * | +0.05 [-0.01, +0.10] | +0 [+0, +0] |
| dev | bare | forward | llada-default − qwen | -3 [-49, +39] | -4 [-29, +16] | +11 [-19, +39] | -14 [-44, +12] | -0.04 [-0.27, +0.14] | +0.07 [+0.02, +0.13] * | +0 [+0, +0] |
| dev | bare | reverse | dream − qwen | -24 [-49, -4] * | -54 [-76, -31] * | +3 [-38, +41] | -22 [-35, -11] * | -0.24 [-0.45, -0.03] * | +0.04 [+0.01, +0.06] * | +0 [+0, +0] |
| dev | bare | reverse | llada − qwen | -5 [-24, +12] | +9 [-6, +25] | -6 [-26, +14] | -18 [-33, -4] * | +0.07 [-0.08, +0.23] | -0.01 [-0.05, +0.03] | +0 [+0, +0] |
| dev | bare | reverse | dream-default − qwen | -28 [-52, -8] * | -60 [-84, -35] * | -4 [-41, +38] | -28 [-50, -7] * | -0.10 [-0.40, +0.21] | +0.03 [+0.01, +0.05] * | +0 [+0, +0] |
| dev | bare | reverse | llada-default − qwen | -8 [-34, +19] | +14 [-14, +40] | -15 [-42, +14] | -6 [-29, +9] | +0.13 [-0.12, +0.36] | +0.02 [-0.01, +0.04] | +0 [+0, +0] |

## 4. The original couplets under the same checker

Each original line scored as if it had been generated from the other. This is the ceiling for the rules.

| Set | Direction | Items | Length | End tone (modern) | End tone (classical) | No repeat | All rules | All rules (classical tone) | Parallelism |
|---|---|---|---|---|---|---|---|---|---|
| dev | forward | 10 | 100% | 60% | 60% | 100% | 60% | 60% | 0.87 |
| dev | reverse | 10 | 100% | 60% | 60% | 100% | 60% | 60% | 0.87 |
