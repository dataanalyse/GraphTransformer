# V3 Plan

`v3` starts as a clean scaffold copied from the stable `v2` workspace.

## Immediate Purpose

- preserve `v2` as the completed paper-facing reference line
- give the next research iteration its own isolated workspace
- allow new modeling and simulation changes without destabilizing `v2`

## Starting Assumption

The current `v3` code is an inherited baseline, not a finalized design. The next step is to adapt the copied scaffold to the new research direction before launching substantive runs.

## Good First Tasks

- define the main `v3` research question
- decide which parts of the `v2` pipeline remain valid
- identify any simulator, feature, target, or architecture changes
- update configs and runner names if the new design diverges materially

## Working Rule

When in doubt:

- experiment in `v3/`
- validate ideas against `v2/`
- keep `v2/` frozen
