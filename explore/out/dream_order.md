# Dream's sampler ranking at step 0

Post hoc; see `PROTOCOL.md` entry 36 and `explore/probe_dream_order.py`. 300 seeded canvases, phase 1 prompt and canvas, Dream's own entropy score after temperature scaling and top-p 0.95 truncation.

| Temperature | First pick in the line | First pick leftmost | Positions tied at the best score (mean) | Line positions with zero entropy (mean) | Padding positions with zero entropy (mean) |
|---|---|---|---|---|---|
| 0.5 | 99% | 47% | 6.8 | 4.5 | 2.2 |
| 0.2 | 100% | 83% | 11.9 | 9.6 | 2.3 |

At padding positions (temperature 1), the end mass going to <|endoftext|> rather than <|im_end|>: mean 1.00 over 684 positions; share of positions where neither takes 95% of it: 0%.

