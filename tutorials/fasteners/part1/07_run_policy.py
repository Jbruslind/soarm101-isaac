"""F7: run your fine-tuned policy closed-loop in Isaac Sim and count how often it places the bolt.

The policy runs in the LeRobot policy server (``scripts/serve_lerobot.sh``); this script is the
robot side. Every control step it sends what the robot sees (both cameras, joint angles, the
prompt) to the server whenever an env has run out of actions, gets the next actions back
(absolute joint targets), and steps the joint-space task with them. Unlike F6, the policy now
sees the result of its own actions, so its mistakes add up: this is the real test.

It prints one line per episode, then ``[F7] PASS`` (or ``[F7] FAIL``: placed in fewer than
``--min-rate`` of the episodes, default 60 %) with the success rate and its 95 % Wilson interval. Run it in the isaac-sim container while the server is running::

    ./scripts/isaac.sh --gui /isaac-sim/python.sh /tutorials/fasteners/part1/07_run_policy.py --episodes 10

Every episode uses a bolt pose from ``--seed`` (default 200: neither the training demos, seed 0,
nor the F6 held-out demos, seed 100). The simulation waits for the server, so the result does
not depend on how fast your GPU is; ``real-time factor`` tells you whether it would keep up live.
"""

# [start:args]
import argparse
import math
import os
import sys
import time

from isaaclab.app import add_launcher_args, launch_simulation

parser = argparse.ArgumentParser(description="F7: run a policy from the policy server closed-loop.")
parser.add_argument("--server", default="ws://127.0.0.1:8000", help="Policy server URL.")
parser.add_argument("--task", default="Soarm-PickPlace-Fastener-JointAbs-v0", help="Joint-space task (contract action).")
parser.add_argument("--episodes", type=int, default=10)
parser.add_argument("--num_envs", type=int, default=1, help="Envs run in parallel; requests are batched.")
parser.add_argument("--seed", type=int, default=200, help="Bolt poses (training used 0, F6 held-out 100).")
parser.add_argument("--n-action-steps", type=int, default=0,
                    help="Actions to run before asking again (0: the server's setting).")
parser.add_argument("--raw-gripper", action="store_true",
                    help="Send the policy's gripper value as is (default: snap it to open/closed, see snap_gripper).")
parser.add_argument("--trace", default="", help="Directory: save each episode's trace (.npz) and a timeline picture.")
parser.add_argument("--min-rate", type=float, default=0.6, help="Pass threshold: fraction of episodes placed.")
add_launcher_args(parser)
args = parser.parse_args()

# scripts/isaac.sh --gui sets HEADLESS=0. Isaac Lab 3.0 opens a window only for a Kit visualizer.
if os.environ.get("HEADLESS") == "0" and not args.visualizer:
    args.visualizer = "kit"
args.enable_cameras = True          # the policy needs both cameras
args.require_kit = True
sys.path.insert(0, os.environ.get("SOARM_ISAAC_ENVS", "/isaac_envs"))
# [end:args]


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95 % Wilson score interval of a success rate k / n."""
    if n == 0:
        return 0.0, 1.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


# [start:gripper]
GRIP_OPEN, GRIP_CLOSED = 1.0, -0.15     # rad: the two gripper targets in every demo (ExpertParams)
CLOSE_BELOW, OPEN_ABOVE = 0.3, 0.55     # hysteresis around the midpoint (0.425 rad)


def snap_gripper(cmd: float, closed: bool) -> tuple[float, bool]:
    """Binary gripper: the demos only ever command open or closed, but a regression policy
    averages near the switch and outputs in-between values, i.e. half-closed jaws that cannot
    hold a bolt. Snap to open/closed, with hysteresis so it does not flicker at the threshold."""
    closed = cmd < CLOSE_BELOW if not closed else cmd < OPEN_ABOVE
    return (GRIP_CLOSED if closed else GRIP_OPEN), closed
# [end:gripper]


# [start:client]
class PolicyClient:
    """Websocket client for the policy server (OpenPi protocol, msgpack-numpy messages)."""

    def __init__(self, url: str):
        from websockets.sync.client import connect

        from soarm_common import wire

        self.wire = wire
        self.ws = connect(url, max_size=None, compression=None, open_timeout=60)
        self.meta = wire.unpackb(self.ws.recv())               # server sends its metadata first

    def infer(self, request: dict) -> dict:
        self.ws.send(self.wire.packb(request))
        reply = self.wire.unpackb(self.ws.recv())
        if "error" in reply:
            raise RuntimeError(f"policy server: {reply['error']}")
        return reply
# [end:client]


# [start:trace]
def save_trace(out: str, episode: int, rows, result: dict, hz: float) -> None:
    """Episode trace (EE height, bolt height, gripper angle and command) as .npz plus a timeline."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    os.makedirs(out, exist_ok=True)
    name = f"F7_episode{episode:02d}_{'placed' if result['success'] else 'failed'}"
    np.savez(os.path.join(out, name + ".npz"), ee_z=rows[:, 0], bolt_z=rows[:, 1], gripper=rows[:, 2],
             gripper_cmd=rows[:, 3], **result)
    t = np.arange(len(rows)) / hz
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(9, 4.6), sharex=True)
    a1.plot(t, 100 * rows[:, 0], label="gripper (EE) height")
    a1.plot(t, 100 * rows[:, 1], label="bolt height")
    a1.set_ylabel("height above table (cm)")
    a1.legend(loc="upper right", fontsize=8)
    a2.plot(t, rows[:, 3], "--", label="gripper command (policy)")
    a2.plot(t, rows[:, 2], label="gripper angle")
    a2.set_ylabel("gripper (rad)")
    a2.set_xlabel("time (s)")
    a2.legend(loc="upper right", fontsize=8)
    fig.suptitle(f"F7 episode {episode}: {'placed' if result['success'] else 'failed'} "
                 f"(grasped={result['grasped']:d} lifted={result['lifted']:d} in_bin={result['in_bin']:d})", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(out, name + ".png"), dpi=100)
    plt.close(fig)
# [end:trace]


def main() -> int:
    with launch_simulation(None, args):
        import gymnasium as gym
        import numpy as np
        import torch
        from isaaclab_tasks.utils import parse_env_cfg

        import soarm_tasks  # noqa: F401  (registers the tasks)
        from soarm_common import contract as C
        from soarm_tasks.mdp import language_instructions

        client = PolicyClient(args.server)
        m = client.meta
        print(f"[F7] server: {m.get('policy_type')} on {m.get('device')}, chunk {m.get('chunk_size')}, "
              f"re-plans every {args.n_action_steps or m.get('n_action_steps')} steps; {m.get('checkpoint')}", flush=True)

        cfg = parse_env_cfg(args.task, num_envs=args.num_envs)
        cfg.seed = args.seed
        env = gym.make(args.task, cfg=cfg)
        u = env.unwrapped
        obs, _ = env.reset(seed=args.seed)
        prompts = language_instructions(u)
        print(f"[F7] task {args.task}, {u.num_envs} env(s), seed {args.seed}, prompt {prompts[0]!r}", flush=True)

        n = u.num_envs
        queues: list[list[np.ndarray]] = [[] for _ in range(n)]
        grip_closed = [False] * n                               # snap_gripper state per env
        quota = [args.episodes // n + (1 if i < args.episodes % n else 0) for i in range(n)]
        done_count = [0] * n
        stages = {k: torch.zeros(n, dtype=torch.bool, device=u.device) for k in ("grasped", "lifted", "in_bin")}
        ep_steps = torch.zeros(n, dtype=torch.long, device=u.device)
        results, latencies, steps, t_start = [], [], 0, time.time()
        trace = [[] for _ in range(n)]                          # per env: one row per step (--trace)
        while len(results) < args.episodes:
            # [start:step]
            need = [i for i in range(n) if not queues[i]]      # envs that ran out of actions
            if need:
                request = {
                    C.POLICY_STATE_KEY: obs["policy"]["joint_pos"][need].cpu().numpy().astype(np.float32),
                    C.POLICY_IMAGES_KEY: {cam.policy_key: obs["images"][cam.obs_key][need].cpu().numpy()
                                          for cam in C.CAMERAS},
                    C.POLICY_PROMPT_KEY: [prompts[i] for i in need],
                }
                if args.n_action_steps:
                    request["n_action_steps"] = args.n_action_steps
                t0 = time.perf_counter()
                actions = client.infer(request)[C.POLICY_ACTIONS_KEY]   # (len(need), H, 6), rad
                latencies.append(time.perf_counter() - t0)
                for j, i in enumerate(need):
                    queues[i] = list(actions[j])
            action = np.stack([q.pop(0) for q in queues])
            if not args.raw_gripper:
                for i in range(n):
                    action[i, 5], grip_closed[i] = snap_gripper(float(action[i, 5]), grip_closed[i])
            action = torch.tensor(action, device=u.device)
            obs, _, terminated, truncated, _ = env.step(action)
            # [end:step]
            steps += 1
            ep_steps += 1
            if args.trace:
                pol = obs["policy"]
                for i in range(n):
                    trace[i].append([float(pol["ee_pose"][i, 2]), float(pol["fastener_pose"][i, 2]),
                                     float(pol["gripper_pos"][i, 0]), float(action[i, 5])])
            for k in stages:
                stages[k] |= obs["subtask_terms"][k].bool().view(-1)
            done = (terminated | truncated).nonzero().squeeze(-1).tolist()
            if done:
                success = u.termination_manager.get_term("success")
                for i in done:
                    queues[i] = []                             # new episode: forget the old plan
                    grip_closed[i] = False
                    if done_count[i] < quota[i]:
                        done_count[i] += 1
                        r = {"success": bool(success[i]), "steps": int(ep_steps[i]),
                             **{k: bool(v[i]) for k, v in stages.items()}}
                        results.append(r)
                        why = "" if r["success"] else ("timed out" if bool(truncated[i]) else "dropped")
                        if args.trace:
                            save_trace(args.trace, len(results), np.array(trace[i]), r, C.CONTROL_HZ)
                        print(f"[F7] episode {len(results):2d} (env {i}): {'PLACED' if r['success'] else 'failed'} "
                              f"in {r['steps'] / C.CONTROL_HZ:.1f} s  grasped={r['grasped']:d} lifted={r['lifted']:d} "
                              f"in_bin={r['in_bin']:d} {why}", flush=True)
                    trace[i] = []
                    for v in stages.values():
                        v[i] = False
                    ep_steps[i] = 0
        wall = time.time() - t_start
        k = sum(r["success"] for r in results)
        lo, hi = wilson(k, len(results))
        lat = np.array(latencies) * 1000
        sim_s = steps / C.CONTROL_HZ
        print(f"[F7] grasped {sum(r['grasped'] for r in results)}/{len(results)}, lifted "
              f"{sum(r['lifted'] for r in results)}/{len(results)}, in bin {sum(r['in_bin'] for r in results)}/{len(results)}")
        print(f"[F7] {len(lat)} policy requests, round trip median {np.median(lat):.0f} ms (p95 "
              f"{np.percentile(lat, 95):.0f} ms); {sim_s:.0f} s simulated in {wall:.0f} s "
              f"(real-time factor {sim_s / wall:.2f})")
        ok = k >= args.min_rate * len(results)
        print(f"[F7] {'PASS' if ok else 'FAIL'} placed {k}/{len(results)} = {100 * k / len(results):.0f}% "
              f"(95% Wilson CI {100 * lo:.0f}-{100 * hi:.0f}%), need >= {100 * args.min_rate:.0f}%", flush=True)
        env.close()
        return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
