#!/usr/bin/env python3
"""Stand-in OpenPi policy server for testing the sim / ROS 2 pipeline without a GPU model.

Speaks the same wire protocol as OpenPi's ``WebsocketPolicyServer`` (what
``openpi_client`` and the soarm_vla_bridge node talk to):

  * on connect the server sends one msgpack-numpy packed metadata dict;
  * then for every packed observation it receives it replies with a packed
    dict ``{"actions": float32 array (horizon, action_dim), ...}``.
  * ``GET /healthz`` returns 200 (same as OpenPi).

Actions are a slow sinusoid in joint space (absolute joint targets, radians), so
the arm visibly moves when the round trip works. Each request is logged with the
received state and per-camera image statistics (a mean of ~0 means a blank frame).

Runs anywhere ``openpi-client`` is installed, e.g. in the ros2-bridge image (from the repo root):

    docker run --rm --network host -v "$PWD/scripts:/scripts:ro" \\
        -v "$PWD/isaac_envs/soarm_common:/opt/soarm/soarm_common:ro" --entrypoint python3 \\
        soarm-ros2 /scripts/mock_policy_server.py --port 8000
    ./scripts/interactive_test.sh --remote 127.0.0.1 --port 8000

See "Testing without a model" in docs/INTERACTIVE_INFERENCE.md.
"""

from __future__ import annotations

import argparse
import asyncio
import http
import importlib
import logging
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
from openpi_client import msgpack_numpy
from websockets.asyncio.server import serve
from websockets.exceptions import ConnectionClosed

log = logging.getLogger("mock_policy")


def load_contract():
    """Import the shared contract (isaac_envs/soarm_common/contract.py): from PYTHONPATH,
    $SOARM_COMMON_DIR, /opt/soarm (docker mount) or the repo checkout this script is in."""
    try:
        return importlib.import_module("soarm_common.contract")
    except ImportError:
        pass
    repo_envs = Path(os.path.realpath(__file__)).parents[1] / "isaac_envs"
    for cand in (os.environ.get("SOARM_COMMON_DIR"), "/opt/soarm", str(repo_envs)):
        if cand and (Path(cand) / "soarm_common" / "contract.py").is_file():
            sys.path.insert(0, cand)
            return importlib.import_module("soarm_common.contract")
    raise ImportError("soarm_common.contract not found; mount isaac_envs/soarm_common at /opt/soarm/soarm_common")


C = load_contract()


def make_actions(t0: float, horizon: int, dim: int, dt: float, amp: float, period: float) -> np.ndarray:
    """(horizon, dim) sinusoidal joint targets; joint j is phase-shifted by j * 60 deg."""
    t = t0 + dt * np.arange(horizon)[:, None]
    phase = np.arange(dim)[None, :] * (math.pi / 3.0)
    return (amp * np.sin(2.0 * math.pi * t / period + phase)).astype(np.float32)


def describe_obs(obs: dict) -> str:
    parts = []
    state = obs.get(C.POLICY_STATE_KEY, obs.get("observation/joint_position"))
    if state is not None:
        parts.append("state=" + np.array2string(np.asarray(state), precision=3, suppress_small=True))
    images = dict(obs.get(C.POLICY_IMAGES_KEY, {}))
    images.update({k: v for k, v in obs.items() if k.startswith("observation/") and "image" in k})
    for name, img in images.items():
        img = np.asarray(img)
        parts.append(f"{name}={img.shape}/{img.dtype} mean={img.mean():.1f} std={img.std():.1f}")
    parts.append(f"prompt={obs.get(C.POLICY_PROMPT_KEY)!r}")
    return " ".join(parts)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--horizon", type=int, default=10, help="actions per chunk")
    p.add_argument("--action-dim", type=int, default=C.ACTION_DIM, help="6 for soarm mode, 8 for droid mode")
    p.add_argument("--dt", type=float, default=C.CONTROL_DT,
                   help="time between actions (contract control rate, 30 Hz)")
    p.add_argument("--amplitude", type=float, default=0.3, help="rad")
    p.add_argument("--period", type=float, default=8.0, help="s")
    p.add_argument("--latency", type=float, default=0.0, help="artificial inference delay, s")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    packer = msgpack_numpy.Packer()
    start = time.monotonic()
    n_requests = 0

    async def handler(ws):
        nonlocal n_requests
        log.info("client connected from %s", ws.remote_address)
        await ws.send(packer.pack({"mock": True, "action_dim": args.action_dim, "horizon": args.horizon}))
        try:
            while True:
                obs = msgpack_numpy.unpackb(await ws.recv())
                n_requests += 1
                t_infer = time.monotonic()
                if args.latency > 0:
                    await asyncio.sleep(args.latency)
                actions = make_actions(
                    time.monotonic() - start, args.horizon, args.action_dim, args.dt, args.amplitude, args.period
                )
                log.info("request #%d %s -> actions %s first=%s", n_requests, describe_obs(obs), actions.shape,
                         np.array2string(actions[0], precision=3))
                await ws.send(packer.pack({
                    C.POLICY_ACTIONS_KEY: actions,
                    "server_timing": {"infer_ms": 1000.0 * (time.monotonic() - t_infer)},
                }))
        except ConnectionClosed:
            log.info("client disconnected")

    def health(connection, request):
        if request.path == "/healthz":
            return connection.respond(http.HTTPStatus.OK, "OK\n")
        return None

    async def run():
        async with serve(handler, args.host, args.port, compression=None, max_size=None,
                         process_request=health) as server:
            log.info("mock policy server listening on ws://%s:%d", args.host, args.port)
            await server.serve_forever()

    asyncio.run(run())


if __name__ == "__main__":
    main()
