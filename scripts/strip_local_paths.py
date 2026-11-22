"""Make the `items_file` recorded in run-file headers relative to the repo.

Usage:
    python scripts/strip_local_paths.py runs/smoke/*.jsonl runs/calib/*.jsonl ...

Run files record which item file they were generated from. Some runs were
started with an absolute path, which puts this machine's user folder into the
file. This rewrites that one field to the path relative to the repo root and
touches nothing else (samples, seeds and settings are left byte for byte).

Never run it on a file that is still being written (the full run appends to
`runs/full/` while it runs).
"""

from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def relative(path: str) -> str:
    p = Path(path)
    if not p.is_absolute():
        return path
    try:
        return p.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return p.name


def main() -> int:
    files = sorted({f for pattern in sys.argv[1:] for f in glob.glob(pattern)})
    changed = 0
    for f in files:
        lines = Path(f).read_text(encoding="utf-8").splitlines(keepends=True)
        out = []
        touched = False
        for line in lines:
            if line.startswith('{"type": "header"'):
                rec = json.loads(line)
                new = relative(rec.get("items_file", ""))
                if new != rec.get("items_file"):
                    rec["items_file"] = new
                    line = json.dumps(rec, ensure_ascii=False) + "\n"
                    touched = True
            out.append(line)
        if touched:
            Path(f).write_text("".join(out), encoding="utf-8", newline="\n")
            changed += 1
    print(f"{changed} of {len(files)} files rewritten")
    return 0


if __name__ == "__main__":
    sys.exit(main())
