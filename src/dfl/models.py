"""Loading the pinned models, one at a time.

`load(key)` returns a `Loaded` bundle; `unload(bundle)` frees the GPU again.

Precision
---------
The weights are 15–16 GB in bf16 and the GPU has 16 GB, so by default the input
embedding table (about 1 GB) stays on the CPU and everything else goes to the
GPU. An embedding table is a lookup: the vectors that reach the first layer are
identical to a full-GPU run, only the lookup happens in CPU memory. The same
arrangement is used for all three models.

`precision="int8"` loads 8-bit weights with bitsandbytes instead. It exists as
the fallback named in PROTOCOL.md and is only to be used for all models alike.
"""

from __future__ import annotations

import gc
import time
from dataclasses import dataclass, field

import torch
from torch import nn

from .registry import MODELS, ModelSpec, local_path


class CpuEmbedding(nn.Module):
    """Looks token ids up in a table kept on the CPU, returns vectors on `device`.

    The table is deliberately *not* registered as a submodule, so that
    `model.device`, `model.to(...)` and `model.parameters()` see only what is
    on the GPU.
    """

    def __init__(self, table: nn.Embedding, device: torch.device):
        super().__init__()
        object.__setattr__(self, "_table", table)
        self.out_device = device

    @property
    def weight(self) -> torch.Tensor:
        return self._table.weight

    @property
    def num_embeddings(self) -> int:
        return self._table.num_embeddings

    @property
    def embedding_dim(self) -> int:
        return self._table.embedding_dim

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self._table(input_ids.to("cpu")).to(self.out_device, non_blocking=True)


def _embedding_slot(model: nn.Module, kind: str) -> tuple[nn.Module, str]:
    """(parent module, attribute name) of the input embedding table."""
    if kind == "llada":
        return model.model.transformer, "wte"
    if kind in ("dream", "ar"):
        return model.model, "embed_tokens"
    raise ValueError(f"no embedding slot known for kind {kind!r}")


def _set_child(parent: nn.Module, name: str, child: nn.Module) -> None:
    if isinstance(parent, nn.ModuleDict):
        parent[name] = child
    else:
        setattr(parent, name, child)


@dataclass
class Loaded:
    spec: ModelSpec
    model: nn.Module
    tokenizer: object
    device: torch.device
    precision: str
    embed_on_cpu: bool
    load_seconds: float
    info: dict = field(default_factory=dict)

    def describe(self) -> dict:
        """Everything about the loaded model that a run record needs."""
        import transformers

        return {
            "key": self.spec.key,
            "repo": self.spec.repo,
            "revision": self.spec.revision,
            "licence": self.spec.licence,
            "kind": self.spec.kind,
            "precision": self.precision,
            "embed_on_cpu": self.embed_on_cpu,
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            **self.info,
        }


def load(key: str, *, precision: str = "bf16", embed_on_cpu: bool = True, device: str = "cuda") -> Loaded:
    """Load one model from its local folder (see scripts/download_models.py)."""
    from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

    spec = MODELS[key]
    path = str(local_path(spec))
    dev = torch.device(device)
    t0 = time.time()

    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=spec.remote_code)
    auto = AutoModelForCausalLM if spec.kind == "ar" else AutoModel

    if precision == "bf16":
        model = auto.from_pretrained(
            path, torch_dtype=torch.bfloat16, trust_remote_code=spec.remote_code, low_cpu_mem_usage=True
        )
        model.eval()
        if embed_on_cpu:
            parent, name = _embedding_slot(model, spec.kind)
            table = getattr(parent, name)
            _set_child(parent, name, nn.Identity())      # keep the table out of .to()
            model.to(dev)
            _set_child(parent, name, CpuEmbedding(table, dev))
        else:
            model.to(dev)
    elif precision == "int8":
        from transformers import BitsAndBytesConfig

        model = auto.from_pretrained(
            path,
            quantization_config=BitsAndBytesConfig(load_in_8bit=True),
            device_map={"": 0},
            trust_remote_code=spec.remote_code,
        )
        model.eval()
        embed_on_cpu = False
    else:
        raise ValueError(f"precision must be 'bf16' or 'int8', got {precision!r}")

    n_params = sum(p.numel() for p in model.parameters())
    if precision == "bf16" and embed_on_cpu:
        n_params += table.weight.numel()
    info = {
        "n_params": n_params,
        "gpu_gib_after_load": round(torch.cuda.memory_allocated() / 2**30, 2) if dev.type == "cuda" else None,
    }
    return Loaded(spec, model, tokenizer, dev, precision, embed_on_cpu, time.time() - t0, info)


def unload(loaded: Loaded) -> None:
    """Drop the model and give the GPU memory back."""
    loaded.model = None
    loaded.tokenizer = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
