# The price of enforcing the end-tone rule at the final slot

Post hoc; see PROTOCOL.md entry 36 and explore/probe_rule_price.py. Price = log P(best allowed character) − log P(best character), in nats; 0 when the best is already allowed. A price of 1 nat means the forced character is e^1 ≈ 2.7 times less likely than the model's favourite.

## qwen

| Set | Direction | Best character already allowed | Mean price (nats) | Median price when not allowed |
|---|---|---|---|---|
| unseen | forward | 81% | -0.47 | -1.41 |
| unseen | reverse | 42% | -1.86 | -3.37 |
| recall | forward | 67% | -1.85 | -4.50 |
| recall | reverse | 41% | -3.69 | -5.19 |
| dev | forward | 68% | -1.56 | -4.22 |
| dev | reverse | 39% | -3.66 | -6.31 |

## llada, prefix = the original line

| Set | Direction | Best character already allowed | Mean price (nats) | Median price when not allowed |
|---|---|---|---|---|
| unseen | forward | 77% | -0.36 | -0.69 |
| unseen | reverse | 56% | -0.69 | -1.44 |
| recall | forward | 60% | -1.15 | -2.19 |
| recall | reverse | 48% | -1.71 | -2.81 |
| dev | forward | 60% | -1.21 | -3.09 |
| dev | reverse | 47% | -1.49 | -2.75 |

## qwen, prefix = its own first sample (bare, right length)

| Set | Direction | Best character already allowed | Mean price (nats) | Median price when not allowed |
|---|---|---|---|---|
| unseen | forward | 79% | -0.55 | -2.25 |
| unseen | reverse | 29% | -3.88 | -4.31 |
| recall | forward | 73% | -1.13 | -3.25 |
| recall | reverse | 36% | -3.33 | -4.44 |

## llada, prefix = its own first sample (bare, right length)

| Set | Direction | Best character already allowed | Mean price (nats) | Median price when not allowed |
|---|---|---|---|---|
| unseen | forward | 50% | -0.97 | -1.91 |
| unseen | reverse | 54% | -0.89 | -1.38 |
| recall | forward | 63% | -1.23 | -2.38 |
| recall | reverse | 41% | -2.00 | -2.75 |

