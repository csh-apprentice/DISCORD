# Third-Party Dependencies

DISCORD depends on the VGGT codebase, which is not vendored into this repository:

- [VGGT](https://github.com/facebookresearch/vggt) — the baseline feed-forward reconstruction model

## Installation

Clone VGGT into this `third_party/` directory and install its requirements:

```bash
git clone https://github.com/facebookresearch/vggt.git third_party/vggt
pip install -r third_party/vggt/requirements.txt
pip install -r third_party/vggt/requirements_demo.txt
```

The expected layout is:

```text
DISCORD/
├── third_party/
│   └── vggt/
```

The helper in [discord3d/third_party.py](../discord3d/third_party.py) adds `third_party/vggt` to `sys.path` at import time. If that directory is missing, it falls back to a sibling checkout at `../vggt` relative to the repository root — this is a convenience for our local research workspace and is not required for a fresh install.