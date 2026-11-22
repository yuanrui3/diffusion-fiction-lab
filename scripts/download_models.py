"""Download the pinned models into a plain local folder, one after another.

Usage:
    python scripts/download_models.py            # the embedder and all three models
    python scripts/download_models.py qwen dream # only these

The files go to `<models dir>/<key>/` as ordinary files (no symlinks: the
Hugging Face cache needs a Windows privilege this machine does not grant).
After a model is complete, a small `dfl_revision.json` is written beside it;
the loader refuses a folder whose recorded commit differs from the registry.

Checks free disk space first and refuses to start without a safety margin.
"""

from __future__ import annotations

import json
import shutil
import sys
import time
from pathlib import Path

from huggingface_hub import snapshot_download

from dfl.registry import EMBEDDER, EMBEDDER_IGNORE, MODELS, TAGGER, models_dir

APPROX_GB = {"llada": 16.1, "dream": 15.3, "qwen": 15.3, "bge-m3": 2.3, "ltp-base2": 0.7}
MARGIN_GB = 20.0


def main(argv: list[str]) -> int:
    wanted = argv or [EMBEDDER.key, TAGGER.key, *MODELS]
    specs = []
    for key in wanted:
        if key == EMBEDDER.key:
            specs.append(EMBEDDER)
        elif key == TAGGER.key:
            specs.append(TAGGER)
        elif key in MODELS:
            specs.append(MODELS[key])
        else:
            print(f"unknown model key: {key}", flush=True)
            return 2

    root = models_dir()
    root.mkdir(parents=True, exist_ok=True)
    todo = [s for s in specs if not (root / s.key / "dfl_revision.json").exists()]
    free_gb = shutil.disk_usage(root).free / 1e9
    need_gb = sum(APPROX_GB[s.key] for s in todo)
    print(f"models dir: {root}\nfree: {free_gb:.0f} GB, need about {need_gb:.0f} GB", flush=True)
    if free_gb < need_gb + MARGIN_GB:
        print("not enough free disk space; nothing downloaded", flush=True)
        return 1

    for spec in specs:
        target = root / spec.key
        marker = target / "dfl_revision.json"
        if marker.exists():
            print(f"--> {spec.repo}: already complete, skipping", flush=True)
            continue
        t0 = time.time()
        print(f"--> {spec.repo} @ {spec.revision[:8]} ({spec.licence})", flush=True)
        snapshot_download(
            repo_id=spec.repo,
            revision=spec.revision,
            local_dir=target,
            ignore_patterns=EMBEDDER_IGNORE if spec.key == EMBEDDER.key else None,
        )
        files = [f for f in target.rglob("*") if f.is_file() and ".cache" not in f.parts]
        size = sum(f.stat().st_size for f in files) / 1e9
        marker.write_text(
            json.dumps({"repo": spec.repo, "revision": spec.revision, "licence": spec.licence,
                        "downloaded": time.strftime("%Y-%m-%d %H:%M:%S"), "size_gb": round(size, 2)},
                       indent=1),
            encoding="utf-8",
        )
        print(f"    done in {time.time() - t0:.0f} s, {size:.2f} GB at {target}", flush=True)
    print("ALL DOWNLOADS COMPLETE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
