#!/usr/bin/env python3
"""Check a running policy server against recorded demos, over the real wire protocol.

Plays the frames of a LeRobot dataset to ``ws://HOST:PORT`` the way a robot would: it asks for
actions, uses the returned ones one per frame, and asks again when they run out. It compares
each used action with the recorded one, so with the F6 held-out set the error must match F6's
``06_check_policy.py`` (same checkpoint, same re-plan interval). It also sends one batched
request (N observations in one message) and checks the reply shape. Prints latency too.

Runs in the soarm-lerobot image, next to a server started with ``scripts/serve_lerobot.sh``::

    cd docker
    docker compose --profile serve-lerobot run --rm lerobot-server python /training/scripts/check_policy_server.py \\
        --dataset /workspace/data/lerobot/fastener_m6_heldout
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
from websockets.sync.client import connect

sys.path.insert(0, os.environ.get("SOARM_ISAAC_ENVS", "/isaac_envs"))
from soarm_common import contract as C  # noqa: E402
from soarm_common import wire  # noqa: E402

from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402


def request(frame: dict) -> dict:
    """One dataset frame -> the contract request a robot / the bridge would send."""
    images = {cam.policy_key: (frame[cam.dataset_key].permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
              for cam in C.CAMERAS}
    return {C.POLICY_STATE_KEY: frame[C.DATASET_STATE_KEY].numpy(), C.POLICY_IMAGES_KEY: images,
            C.POLICY_PROMPT_KEY: frame["task"]}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--url", default="ws://127.0.0.1:8000")
    p.add_argument("--dataset", default="/workspace/data/lerobot/fastener_m6_heldout")
    p.add_argument("--episodes", type=int, default=0, help="first N episodes (0: all)")
    p.add_argument("--batch", type=int, default=4, help="size of the batched-request test (0: skip)")
    p.add_argument("--max-error", type=float, default=0.1, help="pass threshold, mean abs error (rad)")
    args = p.parse_args()

    ds = LeRobotDataset("local/check", root=args.dataset)
    with connect(args.url, max_size=None, compression=None, open_timeout=30) as ws:
        meta = wire.unpackb(ws.recv())
        print(f"[server] {meta}")
        errors, latencies = [], []
        for ep in range(args.episodes or ds.meta.total_episodes):
            start, end = ds.meta.episodes["dataset_from_index"][ep], ds.meta.episodes["dataset_to_index"][ep]
            queue: list[np.ndarray] = []
            for i in range(start, end):
                frame = ds[i]
                if not queue:                                  # out of actions: ask again
                    t0 = time.perf_counter()
                    ws.send(wire.packb(request(frame)))
                    reply = wire.unpackb(ws.recv())
                    latencies.append(time.perf_counter() - t0)
                    if "error" in reply:
                        print(f"[check] FAIL server error: {reply['error']}")
                        return 1
                    queue = list(reply[C.POLICY_ACTIONS_KEY])
                errors.append(np.abs(queue.pop(0) - frame[C.DATASET_ACTION_KEY].numpy()))
        err = np.array(errors)
        lat = np.array(latencies) * 1000
        print(f"[check] {len(err)} frames, {len(lat)} requests: mean abs error {err.mean():.3f} rad "
              f"({np.degrees(err.mean()):.1f} deg); per joint " + " ".join(f"{e:.3f}" for e in err.mean(0)))
        print(f"[check] round trip: median {np.median(lat):.0f} ms, p95 {np.percentile(lat, 95):.0f} ms "
              f"(one request per {meta.get('n_action_steps')} control steps = "
              f"{1000 * meta.get('n_action_steps', 1) / C.CONTROL_HZ:.0f} ms budget)")

        ok_batch = True
        if args.batch:
            frames = [ds[i] for i in range(args.batch)]
            reqs = [request(f) for f in frames]
            batched = {C.POLICY_STATE_KEY: np.stack([r[C.POLICY_STATE_KEY] for r in reqs]),
                       C.POLICY_IMAGES_KEY: {k: np.stack([r[C.POLICY_IMAGES_KEY][k] for r in reqs])
                                             for k in reqs[0][C.POLICY_IMAGES_KEY]},
                       C.POLICY_PROMPT_KEY: [r[C.POLICY_PROMPT_KEY] for r in reqs]}
            t0 = time.perf_counter()
            ws.send(wire.packb(batched))
            reply = wire.unpackb(ws.recv())
            shape = np.asarray(reply.get(C.POLICY_ACTIONS_KEY)).shape
            ok_batch = shape == (args.batch, meta["n_action_steps"], C.ACTION_DIM)
            print(f"[check] batched request of {args.batch}: actions {shape} in "
                  f"{1000 * (time.perf_counter() - t0):.0f} ms -> {'OK' if ok_batch else 'WRONG SHAPE'}")

    ok = err.mean() <= args.max_error and ok_batch
    print(f"[check] {'PASS' if ok else 'FAIL'} error={err.mean():.3f}rad (need <= {args.max_error}) batch={'ok' if ok_batch else 'bad'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
