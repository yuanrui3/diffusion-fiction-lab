# Parallelism against a shuffled baseline

Post hoc; see `PROTOCOL.md` entry 36. *Own*: the frozen parallelism score (LTP base2, share of positions with the same word class) against the line actually given. *Shuffled*: the same answer against another item's given line of the same length and role, from the same set. Seeded sample of up to 800 right-length answers per model and set; 95% bootstrap intervals over items.

| Writer | Set | Answers | Own | Shuffled | Own − shuffled |
|---|---|---|---|---|---|
| Originals | unseen | 82 | 0.79 | 0.49 | +0.30 [+0.23, +0.37] |
| Qwen | unseen | 771 | 0.75 | 0.49 | +0.26 [+0.23, +0.30] |
| LLaDA 0.7 | unseen | 768 | 0.76 | 0.49 | +0.27 [+0.22, +0.31] |
| Dream 0.5 | unseen | 491 | 0.70 | 0.42 | +0.28 [+0.21, +0.33] |
| LLaDA 0 | unseen | 123 | 0.76 | 0.49 | +0.27 [+0.21, +0.33] |
| Dream 0.2 | unseen | 447 | 0.62 | 0.42 | +0.19 [+0.14, +0.25] |
| Originals | recall | 1160 | 0.75 | 0.54 | +0.21 [+0.19, +0.22] |
| Qwen | recall | 800 | 0.78 | 0.53 | +0.25 [+0.23, +0.26] |
| LLaDA 0.7 | recall | 800 | 0.78 | 0.52 | +0.26 [+0.24, +0.28] |
| Dream 0.5 | recall | 800 | 0.62 | 0.46 | +0.16 [+0.14, +0.19] |
| LLaDA 0 | recall | 800 | 0.79 | 0.51 | +0.28 [+0.26, +0.29] |
| Dream 0.2 | recall | 800 | 0.62 | 0.46 | +0.16 [+0.13, +0.18] |
