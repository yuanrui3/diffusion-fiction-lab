# Relevance against copying from the summary

Post hoc; see `PROTOCOL.md` entry 36. With-summary answers. *Overall*: the model's mean relevance minus Qwen's. *Same copy share*: the same difference computed within five bands of copy share (pooled quintiles) and averaged with the pooled band weights. 95% bootstrap intervals over items.

## Unseen set

| Model − Qwen | Overall | Same copy share |
|---|---|---|
| LLaDA 0.7 | +0.061 [+0.045, +0.080] | +0.036 [+0.011, +0.060] |
| LLaDA 0 | +0.063 [+0.043, +0.083] | +0.035 [+0.006, +0.064] |
| Dream 0.5 | +0.022 [+0.006, +0.037] | +0.001 [-0.023, +0.024] |
| Dream 0.2 | +0.042 [+0.025, +0.059] | +0.017 [-0.004, +0.038] |

## Both sets

| Model − Qwen | Overall | Same copy share |
|---|---|---|
| LLaDA 0.7 | +0.054 [+0.041, +0.066] | +0.027 [+0.010, +0.041] |
| LLaDA 0 | +0.059 [+0.046, +0.073] | +0.028 [+0.011, +0.042] |
| Dream 0.5 | +0.018 [+0.007, +0.030] | -0.011 [-0.026, +0.003] |
| Dream 0.2 | +0.044 [+0.031, +0.057] | +0.008 [-0.008, +0.026] |

