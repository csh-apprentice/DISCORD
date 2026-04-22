# Third-Party Dependencies

DISCORD depends on VGGT-related code that is not duplicated inside this cleaned repository.

During development, we resolved these dependencies from sibling checkouts:

- `../RobustVGGT`
- `../vggt`

The helper in [discord3d/third_party.py](../discord3d/third_party.py) first looks for local copies under:

- `third_party/RobustVGGT`
- `third_party/vggt`

and then falls back to the sibling workspace layout above.

The expected local layout is:

```text
DISCORD/
├── third_party/
│   ├── RobustVGGT/
│   └── vggt/
```

If you prefer to keep these checkouts outside the repo, the current code also accepts the research-workspace layout:

```text
../RobustVGGT
../vggt
```

For a public release, the cleanest follow-up is one of:

1. add these dependencies as documented git submodules, or
2. provide explicit installation instructions and import against installed packages.

At the moment, this cleaned repo keeps the dependency resolution lightweight so we can reorganize the codebase without breaking the current research environment.
