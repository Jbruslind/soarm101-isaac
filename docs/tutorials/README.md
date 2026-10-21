# Tutorials

Hands-on, prompted tutorials for the SO-ARM101 in Isaac Sim 6.2 / Isaac Lab 3.0. Each
tutorial is a runnable script plus a page that walks through the code, tells you what output
to expect, and ends with short exercises. The layout follows NVIDIA's Isaac Lab tutorials
(`scripts/tutorials/0X_*` in the Isaac Lab repo): *The Code*, *The Code Explained*,
*The Code Execution*. We add a numeric checkpoint and graded exercises to every page.

New here? Start with [F0: Setup and streaming](fasteners/F0_setup.md). Both tracks begin
with it.

## Tracks

### Track A: real-to-sim (#12)

Bring your own room into Isaac Sim and collect LeRobot data in it. All six are planned.

| # | Tutorial | Status |
|---|----------|--------|
| A1 | Isaac 6.x setup | Same as [F0](fasteners/F0_setup.md) |
| A2 | Load a scanned room | planned (#12) |
| A3 | Add a calibrated D455 | planned (#12) |
| A4 | Noisy depth and how it's validated | planned (#12) |
| A5 | Record in sim and reconstruct with scan-twin | planned (#12) |
| A6 | Collect LeRobot data in your own room | planned (#12) |

### Track B: fastener pick-and-place fine-tuning (#40)

Fine-tune a policy (LeRobot SmolVLA, with ACT as a baseline) so the SO-ARM101 picks metric
fasteners and drops them into bins. Part 1 does one M6×20 hex bolt end to end in the Isaac Sim
GUI over Moonlight. Part 2 scales up with headless Isaac Lab.

**Part 1: one fastener, end to end in the Isaac Sim GUI** (scripts in `tutorials/fasteners/part1/`)

| Page | What you do | Checkpoint | Status |
|------|-------------|------------|--------|
| [F0](fasteners/F0_setup.md) | Setup and streaming (same as Track A, tutorial 1) | `[F0] PASS`, arm visible in Moonlight | **available** |
| F1 | Meet the fasteners: generate an M6×20, do the jaw-fit math | asset fits the jaw opening | planned, needs #30 |
| F2 | Build the work cell: drop test, `contact_offset` and `dt` | bolt settles, no jitter | planned, needs #32, #28 |
| F3 | Grasp physics: contact angle vs squeeze, `solve_articulation_contact_last` | grasp holds through a lift | planned, needs #28 |
| F4 | Scripted pick-and-place | ≥ 16/20 successes | planned, needs #32 |
| [F5](fasteners/F5_record_demos.md) | Record demonstrations: 50 scripted + 5 keyboard teleop, check and replay them | `[F5] PASS`, replay 50/50 | **available** (LeRobot conversion: #34) |
| [F6](fasteners/F6_finetune.md) | Convert to LeRobot, fine-tune ACT and SmolVLA on 12 GB, test on held-out demos | `[F6] PASS` (≤ 0.1 rad held-out error) | **available** |
| F7 | Run your policy in Isaac Sim. **Part 1 milestone** | ≥ 6/10 on M6 | planned, needs #36 |

**Part 2: headless scale-up in Isaac Lab** (scripts in `tutorials/fasteners/part2/`)

| Page | What you do | Checkpoint | Status |
|------|-------------|------------|--------|
| F8 | Going headless: `num_envs`, TiledCamera, VRAM/FPS table | your VRAM/FPS table | planned, needs #38 |
| F9 | Variety: domain randomization and prompts | randomized scenes render | planned, needs #38 |
| F10 | Scale demos with Mimic (~1000 demos) | demo count and success rate | planned, needs #38 |
| F11 | Headless fine-tuning | checkpoint on the larger set | planned, needs #35 |
| F12 | Evaluate like a researcher: success matrix per size/type, ablation | filled-in matrix | planned, needs #37 |
| F13 | (optional) RL expert (state-based PPO) | reward curve | planned, no issue yet |
| F14 | (optional) Toward the real arm | — | planned, needs #16, #17 |

Critical path for the fastener track: #28 → #32 → #33 → #34 → #35 → #36 → #37 → #38.

## Prerequisites

- A Linux host with an NVIDIA RTX GPU (12 GB is enough; this repo is developed on an RTX A2000
  12 GB), driver ≥ 580.95.05, Docker and the NVIDIA Container Toolkit. Versions:
  [ISAAC_VERSIONS.md](../ISAAC_VERSIONS.md).
- The `soarm-isaac-sim:6.2.0.1` image (`cd docker && docker compose build isaac-sim`, about an
  hour the first time) and the robot USD (committed; `./scripts/setup_robot_usd.sh` regenerates it).
- For GUI tutorials: the Sunshine stream host and a [Moonlight](https://moonlight-stream.org/)
  client on your LAN ([SUNSHINE_STREAMING.md](../SUNSHINE_STREAMING.md)).
- Python you can read. Isaac Lab experience helps but isn't required: each page explains the
  Isaac Lab calls it uses.

[F0](fasteners/F0_setup.md) walks through all of this and checks it.

## Conventions

**Where things live**

| What | Where |
|------|-------|
| Pages | `docs/tutorials/fasteners/F<N>_<slug>.md` (Track B), `docs/tutorials/real2sim/` (Track A) |
| Scripts | `tutorials/fasteners/part<P>/<NN>_<name>.py`, `NN` = page number (`F0` → `part1/00_check_setup.py`, `F8` → `part2/08_*.py`). A page with several scripts reuses its `NN`. |
| Screenshots | `docs/tutorials/images/F<N>_<what>.png` |
| Page template | [TEMPLATE.md](TEMPLATE.md) |
| Outputs (datasets, checkpoints, logs) | `/data/tutorials/...` inside the container = `data/tutorials/...` in the repo (git-ignored) |

The `isaac-sim` compose service mounts the repo's `tutorials/` directory read-only at
`/tutorials`, next to `/isaac_envs` and `/robot_description`. Scripts import the repo's
environments from `/isaac_envs`. They write only under `/data`.

**How to run a script.** Always run it through `scripts/isaac.sh` from the repository root:

```bash
# Headless: no window, fastest. Use it for checks, data generation and evaluation.
./scripts/isaac.sh --headless /isaac-sim/python.sh /tutorials/fasteners/part1/00_check_setup.py

# GUI: a normal Isaac Sim window on the Sunshine display. Watch it in Moonlight.
./scripts/sunshine.sh up        # once; it keeps running
./scripts/isaac.sh --gui /isaac-sim/python.sh /tutorials/fasteners/part1/00_check_setup.py
```

`--gui` sets `HEADLESS=0`, and tutorial scripts then open the Kit visualizer (Isaac Lab 3.0
opens a window only for `--viz kit`). Extra arguments after the script path go to the script,
for example `... 00_check_setup.py --steps 300`. `./scripts/isaac.sh --help` lists the options.

**What scripts print.** Each script prefixes its own log lines with the page tag (`[F0]`,
`[F4]`, ...) and ends with one `[F<N>] PASS ...` or `[F<N>] FAIL ...` line that holds the
page's checkpoint numbers, then exits 0 or 1. To find it in Kit's log noise:
`... 2>&1 | grep '\[F0\]'`.

**Code on pages.** Pages quote scripts in named regions (`# [start:name]` ... `# [end:name]`)
instead of line numbers, and mark each excerpt with `<!-- snippet: path#name -->`.
`./scripts/check_tutorials.sh` fails if an excerpt no longer matches its script, if a script
does not compile, or if a relative link is broken. Run it before committing a tutorial change.
`scripts/smoke_test.sh` runs it too, and it becomes a CI step once the repo has CI (#3).

**One GPU.** The 12 GB card runs one heavy job at a time. Don't train while Isaac Sim is
running, and close GUI sessions before long headless runs.
