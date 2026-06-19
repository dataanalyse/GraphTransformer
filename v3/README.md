# V3 Workspace

`v3` is the new research workspace built from the stable `v2` scaffold.

The intent is:

- keep `v2/` frozen as the completed second research line
- use `v3/` for the next design iteration and new experiments
- record major design shifts in the root `SESSION_NOTES.md`

For a clean collaborator-facing run guide, see:

- `v3/SETUP_AND_RUN.md`

Current scaffold status:

- the core `v2` codebase has been copied here as a starting point
- configs under `v3/configs/` are inherited starter configs, not final `v3` assumptions
- `v3/data/` and `v3/runs/` are intentionally empty working directories
- `v3` should be treated as the active experimentation lane from this point onward

Recommended practice:

- keep all new code, configs, data, and runs inside `v3/`
- preserve `v2/` as the reproducible reference line for the finished paper package
- promote only stable `v3` outputs into paper-facing artifacts later
