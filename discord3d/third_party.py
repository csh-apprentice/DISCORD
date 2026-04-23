from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _candidate_roots() -> list[Path]:
    repo_root = REPO_ROOT
    workspace_root = repo_root.parent
    return [
        repo_root / "third_party" / "vggt",
        workspace_root / "vggt",
    ]


def setup_vggt_paths() -> None:
    candidates = _candidate_roots()
    existing = [p for p in candidates if p.exists()]
    # aggregator implementation remains the active one.
    for path in reversed([str(p) for p in existing]):
        if path not in sys.path:
            sys.path.insert(0, path)


def resolve_vggt_root() -> Path | None:
    for path in _candidate_roots():
        if path.exists():
            return path
    return None

