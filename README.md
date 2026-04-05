# GraphTransformer Repo

This repository now has two tracks:

- `v1_archive/`: the completed first research line on synthetic supply-chain disruption prediction
- `v2/`: the clean workspace for the next, more graph-centric direction

Repo-level continuity files stay at the root:

- `AGENTS.md`
- `SESSION_NOTES.md`

## Working Conventions

- Use `v1_archive/` when you want to revisit, rerun, or cite the earlier baseline/GCN/transformer experiments.
- Use `v2/` for all new development going forward.
- Update `SESSION_NOTES.md` after meaningful progress so a new session can recover context quickly.

## Revisit V1

To run the archived project again:

```bash
cd v1_archive
../.venv/bin/python run_experiments.py --config configs/experiments.yaml
```

## V1 Research Docs

The v1 paper and experiment documents now live under `v1_archive/`:

- `v1_archive/EXPERIMENT_LOG.md`
- `v1_archive/PAPER_PREP.md`
- `v1_archive/DRAFT_PAPER.md`
- `v1_archive/RESEARCH_STATUS.md`
