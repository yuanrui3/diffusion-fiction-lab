# explore/: post hoc analyses and probes

Everything here was written after the phase 1 results were seen (`PROTOCOL.md` entries 36–37). It reads the frozen outputs and never changes them. None of it is part of the frozen plan, and none of it may be cited as a pre-registered result. The reading of these numbers is in `results/theory.md`.

Run from the repo root in Git Bash, with `export PYTHONUTF8=1`. GPU scripts load one model per process; never run them while a generation is running.

| Script | Needs | Writes | What it asks |
|---|---|---|---|
| `analyse_existing.py` | CPU | `out/existing_data.md`, `.json` | How the final tone is chosen (rule, imitation or bias); what the originals keep; length errors and the canvas; the order the diffusion samplers fill the canvas; copying from the summary |
| `parallel_baseline.py` | CPU (LTP) | `out/parallel_baseline.md`, `.json` | Parallelism against the given line, compared with a shuffled given line |
| `recall_direction.py` | CPU | `out/recall_direction.md` | Exact and partial recall of the classical 回目, by item and direction |
| `probe_tones.py <model>` | GPU | `out/probe_K_<model>.jsonl`, `out/probe_S_<model>.jsonl` | K: does the model know a character's tone? S: does stating the tone rule move its choice of the last character? |
| `report_probes.py` | CPU | `out/probes.md`, `.json` | Tables for probes K and S |
| `probe_length.py <model>` | GPU | `out/probe_L_<model>.jsonl` | Before the first denoising step: where does the model expect the line to end, and which position would its sampler fix first? |
| `report_length.py` | CPU | `out/length_probe.md` | Tables for probe L |
| `probe_checks.py <model>` | GPU | `out/probe_K2_<model>.jsonl` (sanity lines printed) | K2: the tone question with the reply started with 第; sanity check of the final-slot canvas on well-known couplets |
| `probe_S_masked_tail.py <model>` | GPU | `out/probe_S_<model>_maskedtail.jsonl` | Probe S with the rest of the canvas masked instead of padded (the state Dream is actually in) |
| `probe_dream_order.py` | GPU | `out/dream_order.md`, `out/probe_dream_order.jsonl` | Why Dream fills left to right: its sampler's own ranking at step 0 (temperature, top-p, bf16), plus a check of its one-position shift |
| `rescore_opposition.py` | CPU | `out/opposition_rescore.md`, `.json` | Phase 1 all-rules rates with tone opposition in place of tone assignment; the author's grades against copying from the summary |
| `relevance_vs_copy.py` | CPU | `out/relevance_vs_copy.md` | The diffusion models' relevance edge over Qwen, overall and at the same share of characters copied from the summary |
| `copy_first.py` | CPU | `out/copy_first.md` | Do the diffusion samplers fix tokens copied from the summary before the line's other tokens? (Commit rank, also within each third of the line) |
| `probe_rule_price.py <model> [own]` | GPU | `out/rule_price.md` (appends) | What enforcing the end-tone rule at the final slot would cost: how often the model's favourite final character is already allowed, and how much less likely the best allowed one is. `own` uses the model's own first sample as the line instead of the original |

Probes are single forward passes: they read a model's output distribution and sample nothing.
