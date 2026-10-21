# Fastener tutorial scripts

Runnable scripts for the fastener pick-and-place fine-tuning track (#40). The pages that
explain them are in [docs/tutorials/fasteners/](../../docs/tutorials/fasteners/). Start with the
[tutorial index](../../docs/tutorials/README.md).

| Directory | Pages | Mode |
|-----------|-------|------|
| [part1/](part1/) | F0–F7: one M6×20 bolt, end to end | Isaac Sim GUI over Moonlight (headless works too) |
| [part2/](part2/) | F8–F14: headless scale-up in Isaac Lab | headless |

Scripts are named `<NN>_<name>.py`, where `NN` is the page number
(`part1/00_check_setup.py` belongs to [F0](../../docs/tutorials/fasteners/F0_setup.md)).

Run them from the repository root through `scripts/isaac.sh`. The `isaac-sim` service mounts
this directory read-only at `/tutorials`:

```bash
./scripts/isaac.sh --headless /isaac-sim/python.sh /tutorials/fasteners/part1/00_check_setup.py
./scripts/isaac.sh --gui      /isaac-sim/python.sh /tutorials/fasteners/part1/00_check_setup.py
```

Scripts write outputs only under `/data` (`data/` in the repo). Code quoted on the pages sits
between `# [start:<name>]` / `# [end:<name>]` comments. Keep those markers, and run
`./scripts/check_tutorials.sh` after editing a script.
