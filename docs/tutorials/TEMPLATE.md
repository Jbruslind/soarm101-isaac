<!--
Page template for docs/tutorials/. Copy it to docs/tutorials/fasteners/F<N>_<slug>.md,
replace every <...>, and delete these comments. The structure follows NVIDIA's Isaac Lab
tutorials (The Code / The Code Explained / The Code Execution) and adds a numeric checkpoint
and graded exercises. Keep the section order and headings: learners skim by them.
See docs/tutorials/fasteners/F0_setup.md for a finished page. Links below are written
relative to docs/tutorials/fasteners/, where the copy will live (check_tutorials.sh skips
this file because its links only resolve from there).

Before committing, run ./scripts/check_tutorials.sh (scripts compile, links resolve,
snippets match their scripts).
-->

# F<N>: <Title, a verb phrase: "Build the work cell">

> **Track:** <Fasteners Part 1 | Part 2 | Real-to-sim> · **Time:** <~N min> ·
> **GPU:** <GUI over Moonlight | headless> · **Script:** [`tutorials/fasteners/part<P>/<NN>_<name>.py`](../../../tutorials/fasteners/part<P>/<NN>_<name>.py)

<One paragraph: what the learner builds and why it matters for picking fasteners. Name the
Isaac Lab / Isaac Sim concepts it introduces. Link the issue the page depends on (#NN).>

## What you'll learn

- <A concrete skill, e.g. "spawn a rigid fastener with a convex-hull collider">
- <...>
- <2–5 bullets; each one is checked later in Expected output or Try it>

## Prerequisites

- Finished: [F<N-1>](F<N-1>_<slug>.md) (<what it left behind, e.g. "the work-cell USD in data/tutorials/">)
- Running: <e.g. Sunshine stream host (`./scripts/sunshine.sh up`) for GUI steps>
- Background reading (optional): <Isaac Lab doc page or repo doc, e.g. ../../ISAAC_VERSIONS.md>

## The code

The tutorial corresponds to
[`tutorials/fasteners/part<P>/<NN>_<name>.py`](../../../tutorials/fasteners/part<P>/<NN>_<name>.py).
Read it once top to bottom before the walkthrough.

<details>
<summary>Code for <NN>_<name>.py</summary>

<!--
Paste the whole script for short scripts (< ~120 lines). For longer ones, link the file and
quote only the regions below. Never quote by line number: line numbers drift.
-->

```python
<full script>
```

</details>

## The code explained

<!--
One subsection per step, in execution order. Each one: 1–3 short paragraphs on WHY, then the
excerpt. Mark the step's code in the script with
    # [start:<region>]
    ...
    # [end:<region>]
and quote it with the snippet marker below on the line right before the fence;
check_tutorials.sh compares the excerpt with the region and fails when they differ.
For links to a precise line (e.g. in Isaac Lab or a PR), use a GitHub permalink with a
commit hash and #L<a>-L<b>, never a branch name.
-->

### <Step 1: e.g. Launching the simulator>

<Why this step exists, what the key call does, which config field matters.>

<!-- snippet: tutorials/fasteners/part<P>/<NN>_<name>.py#<region> -->
```python
<exact contents of the region>
```

> **Note:** <Isaac Lab 3.0 gotchas belong in notes like this: xyzw quaternions, `.torch`
> on ProxyArray, `env_[^/]+` prim regexes. Link ../../ISAAC_VERSIONS.md.>

### <Step 2: ...>

<...>

## Run it

Run from the repository root. Both modes run the same script.

**GUI (watch in Moonlight).** Start the stream host once (`./scripts/sunshine.sh up`), open
**SO-ARM Isaac Desktop** in Moonlight, then:

```bash
./scripts/isaac.sh --gui /isaac-sim/python.sh /tutorials/fasteners/part<P>/<NN>_<name>.py <args>
```

**Headless (fast, no window).**

```bash
./scripts/isaac.sh --headless /isaac-sim/python.sh /tutorials/fasteners/part<P>/<NN>_<name>.py <args>
```

<Arguments the learner will change: `--flag` (default, meaning).>
<Expected wall time on a 12 GB RTX card, and that the first Kit start takes 2–5 min.>

## Expected output

<What success looks like, in three forms.>

**Log.** Filter the run with `2>&1 | grep '\[F<N>\]'`:

```text
[F<N>] <info line>
[F<N>] PASS <metric>=<value> ...
```

**Screenshot.** <What is on screen and where to look.>

![<alt text: what the screenshot shows>](../images/F<N>_<what>.png)

**Checkpoint.** <The number the learner verifies, with its tolerance, e.g. "success ≥ 16/20",
"bolt rests at z = 0.003 ± 0.001 m after 2 s". Say what to do if it's off: usually see
Troubleshooting.>

## Try it

<!--
2–4 graded exercises: warm-up (change one parameter, predict, observe), core (modify code),
stretch (open-ended). Each states a measurable goal. Answers are hidden in <details>.
Exercises must not require editing files under isaac_envs/: copy the script to
data/tutorials/ (git-ignored) or pass flags.
-->

1. **<Warm-up: predict then observe>.** <Change X to Y. Before running, predict Z. Run it
   and compare.>

   <details><summary>Answer</summary>

   <Expected observation and the reason, with numbers.>

   </details>

2. **<Core: modify the code>.** <Task with a measurable goal.>

   <details><summary>Hint</summary>

   <A pointer, not the solution.>

   </details>

   <details><summary>Answer</summary>

   <Solution sketch or diff, and the expected number.>

   </details>

3. **<Stretch (optional)>.** <Open-ended question tied to the next page.>

   <details><summary>Discussion</summary>

   <What a good answer covers.>

   </details>

## Troubleshooting

| Symptom | Cause / fix |
|---------|-------------|
| <exact error text or behaviour> | <cause, then the fix command> |
| <...> | <...> |

General problems (Kit start, carb shared memory, Moonlight black screen) are in
[F0's troubleshooting](F0_setup.md#troubleshooting) and
[TROUBLESHOOTING.md](../../TROUBLESHOOTING.md).

## Next

<One sentence on what the learner can now do, then the link.>
Next: [F<N+1>: <title>](F<N+1>_<slug>.md). Back to the [tutorial index](../README.md).
