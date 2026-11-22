"""The models this project uses, pinned to exact commits.

Verified on the Hugging Face API on 2026-11-19 (see PROTOCOL.md, entry 3).
Always load with `revision=spec.revision` so that an upstream change cannot
alter a run.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path


def models_dir() -> Path:
    """Folder that holds the downloaded weights, one subfolder per model key.

    Defaults to `_models` beside the repo (outside it, so weights never land in
    git). Override with the DFL_MODELS_DIR environment variable.
    """
    env = os.environ.get("DFL_MODELS_DIR")
    if env:
        return Path(env)
    return Path(__file__).resolve().parents[3] / "_models"


def local_path(spec: "ModelSpec") -> Path:
    """Local folder of a downloaded model, checked against the pinned commit."""
    path = models_dir() / spec.key
    marker = path / "dfl_revision.json"
    if not marker.exists():
        raise FileNotFoundError(
            f"{spec.repo} is not downloaded (no {marker}). Run scripts/download_models.py {spec.key}"
        )
    recorded = json.loads(marker.read_text(encoding="utf-8"))["revision"]
    if recorded != spec.revision:
        raise RuntimeError(
            f"{path} holds commit {recorded[:8]}, but the registry pins {spec.revision[:8]}"
        )
    return path


@dataclass(frozen=True)
class ModelSpec:
    key: str            # short name used in file names and tables
    repo: str           # Hugging Face ID
    revision: str       # commit hash
    kind: str           # "ar", "llada" or "dream": which generation code path
    licence: str
    remote_code: bool   # needs trust_remote_code=True


MODELS: dict[str, ModelSpec] = {
    "llada": ModelSpec(
        key="llada",
        repo="GSAI-ML/LLaDA-8B-Instruct",
        revision="08b83a6feb34df1a6011b80c3c00c7563e963b07",
        kind="llada",
        licence="MIT",
        remote_code=True,
    ),
    "dream": ModelSpec(
        key="dream",
        repo="Dream-org/Dream-v0-Instruct-7B",
        revision="05334cb9faaf763692dcf9d8737c642be2b2a6ae",
        kind="dream",
        licence="Apache-2.0",
        remote_code=True,
    ),
    "qwen": ModelSpec(
        key="qwen",
        repo="Qwen/Qwen2.5-7B-Instruct",
        revision="a09a35458c702b33eeacc393d103063234e8bc28",
        kind="ar",
        licence="Apache-2.0",
        remote_code=False,
    ),
}

EMBEDDER = ModelSpec(
    key="bge-m3",
    repo="BAAI/bge-m3",
    revision="5617a9f61b028005a4858fdac845db406aefb181",
    kind="embedder",
    licence="MIT",
    remote_code=False,
)

# Files of bge-m3 that the PyTorch path does not need (ONNX export, images).
EMBEDDER_IGNORE = ["onnx/*", "imgs/*", "*.jpg", "*.webp", ".DS_Store"]

# Word segmentation and part-of-speech tagging for the parallelism score.
# Chosen over jieba and Stanza in scripts/compare_taggers.py (PROTOCOL.md, entry 25).
# LTP is free for research and personal use; commercial use needs a licence from HIT.
TAGGER = ModelSpec(
    key="ltp-base2",
    repo="LTP/base2",
    revision="70c57011dab606d130c4027705fa155118ca396f",
    kind="tagger",
    licence="LTP licence (free for research and personal use)",
    remote_code=False,
)
