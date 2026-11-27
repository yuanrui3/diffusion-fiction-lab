# Commit order against copying

Post hoc; see `PROTOCOL.md` entry 36. With-summary samples, both sets. Each token of the line gets its rank in the line's commit order (0 = fixed first among the line's tokens, 1 = last). Tokens are split by whether every 汉字 in them occurs in the chapter summary. Tokens with no 汉字 are left out.

| Sampler | Tokens from the summary | Their mean rank | Other tokens | Their mean rank | Difference |
|---|---|---|---|---|---|
| LLaDA 0.7 | 6038 | 0.43 | 2394 | 0.68 | -0.25 |
| LLaDA 0 | 782 | 0.44 | 280 | 0.69 | -0.25 |
| Dream 0.5 | 6463 | 0.45 | 1741 | 0.68 | -0.23 |
| Dream 0.2 | 9186 | 0.47 | 1983 | 0.64 | -0.17 |

Controlled for position (the difference computed within the first, middle and last third of the line, then averaged):

| Sampler | Difference within position |
|---|---|
| LLaDA 0.7 | -0.22 |
| LLaDA 0 | -0.23 |
| Dream 0.5 | -0.19 |
| Dream 0.2 | -0.09 |

