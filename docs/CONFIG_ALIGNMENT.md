# SOARM101 OpenPI Configuration Alignment

This guide prevents schema mismatches between Isaac Sim, ROS2 bridge, and OpenPI
policies. Most "robot moves back-and-forth" failures are configuration mismatches,
not raw motion-control issues.

## 1) Choose One Explicit Policy Mode

Do not mix modes or rely on implicit fallback.

| Mode | `OPENPI_POLICY_MODE` | Policy config | Checkpoint requirement | Observation contract | Action contract |
|---|---|---|---|---|---|
| SOARM-native | `soarm` | `soarm_*` (e.g. `soarm_pi0_fast`) | Required, local SOARM checkpoint dir | `state` (6D), `images.cam_wrist`, `images.cam_overhead` | 6D absolute SOARM joint targets |
| DROID-compat | `droid` | OpenPI DROID config (e.g. `pi0_fast_droid`) or `--env DROID` | Optional | `observation/joint_position` (7D padded), `observation/gripper_position` (1D), wrist/exterior images | 8D DROID action remapped to SOARM 6D |

### Fail-fast behavior now enforced

- `docker/openpi-server/entrypoint.sh` rejects:
  - `OPENPI_POLICY_MODE=soarm` with non-`soarm_*` config
  - `OPENPI_POLICY_MODE=soarm` without a valid checkpoint directory
  - `OPENPI_POLICY_MODE=droid` with `soarm_*` config
- There is no silent fallback from SOARM to DROID.

## 2) Launch Examples

### SOARM-native (recommended)

```bash
./scripts/interactive_test.sh \
  --policy-mode soarm \
  --model-config soarm_pi0_fast \
  --checkpoint /path/to/soarm_lora
```

### DROID-compat (fallback/demo)

```bash
./scripts/interactive_test.sh \
  --policy-mode droid \
  --model-config pi0_fast_droid
```

## 3) Preflight Checklist (Block Before Execute)

The bridge now validates the following before first inference:

1. `/joint_states` has samples.
2. Expected SOARM joint names exist.
3. Wrist image is present (`OPENPI_REQUIRE_WRIST_IMAGE=1` by default).
4. Overhead image present if required (`OPENPI_REQUIRE_OVERHEAD_IMAGE=1`).
5. Policy mode is valid (`soarm` or `droid`).

When checks pass, logs show:

`[OpenVLA] Preflight OK: joint/image contracts satisfied; starting inference.`

## 4) Camera and Joint Contract

Canonical SOARM joint order (bridge + observations):

`shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper`

Camera topics consumed by bridge:

- `/camera/wrist/image_raw` (required by default)
- `/camera/overhead/image_raw` (optional by default)

## 5) Hardware and Deployment Recommendations

### Stable deployment hierarchy

1. **Best stability:** Isaac Sim + ROS2 bridge on RTX workstation, OpenPI on remote GPU.
2. **Single-machine dev:** RTX 12-16 GB VRAM class minimum for smooth interactive work.
3. **Jetson edge:** Use DROID mode unless you have validated SOARM checkpoint/config support.

### Practical sizing

- Inference-only practical floor: **8-12 GB VRAM**.
- Fine-tuning (LoRA/full): substantially larger VRAM (see OpenPI requirements).
- Isaac livestream host needs **NVENC** (use RTX host for visualization).

## 6) External References

- OpenPI repository and model docs: <https://github.com/Physical-Intelligence/openpi>
- Physical Intelligence research updates: <https://www.pi.website/>
- SOARM Isaac Lab project reference: <https://github.com/MuammerBay/isaac_so_arm101>
