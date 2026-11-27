# What the diffusion models expect before the first step

Post hoc; see `PROTOCOL.md` entry 36 and `explore/probe_length.py`. One forward pass on the fully masked phase 1 canvas (required characters + 6 tokens), exact phase 1 prompt, bare, both directions, all 723 items.

*Expected length*: the first canvas position whose probability of an end token exceeds 0.5. *Original's length*: the original line in the model's own tokens. *First pick*: the position each sampler would fix first: for LLaDA (low-confidence remasking) the most confident position, for Dream (entropy) the lowest-entropy position. *In the padding* means that position is predicted to be an end token. *Leftmost* means it is the first canvas position.

| Model | Canvases | Expected length = original's | Mean expected − original (tokens) | First pick in the padding | First pick leftmost | p(end), line part | p(end), padding part |
|---|---|---|---|---|---|---|---|
| LLaDA | 1446 | 40% | -0.09 | 100% | 0% | 0.05 | 0.94 |
| Dream | 1446 | 2% | +3.78 | 100% | 0% | 0.09 | 0.82 |

