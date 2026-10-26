#!/usr/bin/env python3
"""Serve a fine-tuned LeRobot policy (ACT or SmolVLA) on the OpenPi websocket protocol.

Drop-in replacement for ``scripts/mock_policy_server.py`` and the OpenPi server: the ros2-bridge
(``soarm_vla_bridge``), ``interactive_inference.py`` and the Isaac Sim tutorial client all talk
to it unchanged. Protocol (see ``isaac_envs/soarm_common/wire.py``):

* on connect: one packed metadata dict (policy type, chunk size, action dim, control rate, ...);
* request: ``{"state": (6,), "images": {"cam_wrist": HxWx3 uint8, "cam_overhead": ...},
  "prompt": str}``; batched requests (state ``(N, 6)``, images ``(N, H, W, 3)``, prompt a list)
  get ``(N, H, 6)`` back, so an evaluation can query many envs in one call; an optional
  ``"n_action_steps"`` overrides the server's default for that request;
* reply: ``{"actions": float32 (n_action_steps, 6) absolute joint targets in rad, "server_timing": {...}}``;
* ``GET /healthz`` returns 200.

Each request is answered from scratch (``predict_action_chunk``), so the server keeps no
per-client state: the client runs the returned actions and asks again when it runs out. The
number of actions returned is how often the policy re-plans (F6: re-planning every 10 steps cut
the held-out error 2.5x compared with running a whole 100-step ACT chunk).

Runs in the soarm-lerobot image; ``scripts/serve_lerobot.sh`` starts it with the right mounts.
"""

from __future__ import annotations

import argparse
import asyncio
import http
import json
import logging
import os
import sys
import threading
import time

import numpy as np
import torch
from websockets.asyncio.server import serve
from websockets.exceptions import ConnectionClosed

sys.path.insert(0, os.environ.get("SOARM_ISAAC_ENVS", "/isaac_envs"))
from soarm_common import contract as C  # noqa: E402
from soarm_common import wire  # noqa: E402

from lerobot.policies.factory import get_policy_class, make_pre_post_processors  # noqa: E402

log = logging.getLogger("lerobot_policy_server")

# Rough inference memory of one policy process (CUDA context included), used to decide whether
# the GPU has room or the server should fall back to the CPU.
INFER_GB = {"act": 0.8, "smolvla": 1.5}


def model_dir(path: str) -> str:
    """Accept a run dir (data/checkpoints/<job>) as well as its pretrained_model dir."""
    last = os.path.join(path, "checkpoints", "last", "pretrained_model")
    return last if os.path.isdir(last) else path


def pick_device(requested: str, policy_type: str) -> str:
    """"auto": the GPU if it has room for the policy next to whatever already runs there, else CPU."""
    if requested != "auto":
        return requested
    if not torch.cuda.is_available():
        return "cpu"
    free, total = torch.cuda.mem_get_info()
    need = INFER_GB.get(policy_type, 2.0) * 1024**3
    if free < need:
        log.warning("GPU has %.1f GB free of %.1f GB, the %s policy needs about %.1f GB: serving on CPU",
                    free / 1024**3, total / 1024**3, policy_type, need / 1024**3)
        return "cpu"
    return "cuda"


# [start:load]
class LeRobotPolicy:
    """A checkpoint plus its saved processors, answering contract observations with action chunks."""

    def __init__(self, checkpoint: str, device: str, n_action_steps: int, flow_steps: int = 0):
        path = model_dir(checkpoint)
        with open(os.path.join(path, "config.json")) as f:
            self.policy_type = json.load(f)["type"]
        self.device = pick_device(device, self.policy_type)
        self.policy = get_policy_class(self.policy_type).from_pretrained(path).to(self.device).eval()
        if flow_steps and hasattr(self.policy.config, "num_steps"):
            self.policy.config.num_steps = flow_steps          # SmolVLA: fewer flow steps, faster chunks
        with open(os.path.join(path, "train_config.json")) as f:
            rename_map = json.load(f).get("rename_map") or {}
        self.pre, self.post = make_pre_post_processors(self.policy.config, pretrained_path=path, preprocessor_overrides={
            "device_processor": {"device": self.device},
            "rename_observations_processor": {"rename_map": rename_map}})
        self.chunk_size = int(self.policy.config.chunk_size)
        self.n_action_steps = min(n_action_steps or self.policy.config.n_action_steps, self.chunk_size)
        # Images are fed at the contract camera size, as in the training data. (The policy's own
        # input_features can say otherwise, e.g. 256x256 inherited from smolvla_base; the policy
        # resizes internally, exactly as it did in training.)
        self.image_hw = {cam.policy_key: (cam.height, cam.width) for cam in C.CAMERAS}
        self.checkpoint = path
        self.lock = threading.Lock()                           # one inference at a time on the device
# [end:load]

    def metadata(self) -> dict:
        return {"server": "lerobot", "policy_type": self.policy_type, "checkpoint": self.checkpoint,
                "device": self.device, "chunk_size": self.chunk_size, "n_action_steps": self.n_action_steps,
                "action_dim": C.ACTION_DIM, "control_hz": C.CONTROL_HZ, "contract_version": C.CONTRACT_VERSION,
                "image_hw": {k: list(v) for k, v in self.image_hw.items()}, "joint_names": list(C.JOINT_NAMES),
                "cameras": [cam.policy_key for cam in C.CAMERAS]}

    # [start:to_batch]
    def to_lerobot(self, obs: dict) -> tuple[dict, bool]:
        """Contract request -> LeRobot batch: state, one float CHW image per camera, task strings."""
        state = np.asarray(obs[C.POLICY_STATE_KEY], dtype=np.float32)
        batched = state.ndim == 2
        state = state if batched else state[None]
        n = state.shape[0]
        batch = {C.DATASET_STATE_KEY: torch.from_numpy(state)}
        images = obs.get(C.POLICY_IMAGES_KEY, {})
        for cam in C.CAMERAS:                                  # cam_wrist -> observation.images.wrist, ...
            if cam.policy_key not in images:
                raise KeyError(f"missing image {cam.policy_key!r} (got {sorted(images)})")
            img = np.asarray(images[cam.policy_key])
            img = img if img.ndim == 4 else img[None]
            x = torch.from_numpy(np.ascontiguousarray(img)).permute(0, 3, 1, 2).float() / 255.0
            hw = self.image_hw[cam.policy_key]
            if tuple(x.shape[-2:]) != hw:                      # e.g. a real camera at another size
                x = torch.nn.functional.interpolate(x, size=hw, mode="bilinear", align_corners=False)
            batch[cam.dataset_key] = x
        prompt = obs.get(C.POLICY_PROMPT_KEY) or ""
        batch["task"] = list(prompt) if isinstance(prompt, (list, tuple)) else [str(prompt)] * n
        return batch, batched
    # [end:to_batch]

    # [start:infer]
    @torch.no_grad()
    def infer(self, obs: dict) -> dict:
        batch, batched = self.to_lerobot(obs)
        n = int(obs.get("n_action_steps") or self.n_action_steps)
        n = max(1, min(n, self.chunk_size))
        with self.lock:
            t0 = time.perf_counter()
            self.policy.reset()                                # stateless: no queue carried over
            chunk = self.policy.predict_action_chunk(self.pre(batch))   # (N, chunk_size, 6), normalized
            actions = self.post(chunk)                         # -> absolute joint targets, rad
            infer_ms = 1000 * (time.perf_counter() - t0)
        actions = actions[:, :n].float().cpu().numpy()
        return {C.POLICY_ACTIONS_KEY: actions if batched else actions[0],
                "server_timing": {"infer_ms": infer_ms, "batch": int(actions.shape[0]), "n_action_steps": n}}
    # [end:infer]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--checkpoint", required=True, help="data/checkpoints/<job> or a pretrained_model dir")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--device", default="auto", help="auto (GPU if it has room, else CPU), cuda, cuda:1, cpu")
    p.add_argument("--n-action-steps", type=int, default=10,
                   help="actions returned per request (how often the policy re-plans); 0: the trained value")
    p.add_argument("--flow-steps", type=int, default=0, help="SmolVLA denoising steps (0: trained value, 10)")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s", force=True)

    t0 = time.time()
    pol = LeRobotPolicy(args.checkpoint, args.device, args.n_action_steps, args.flow_steps)
    meta = pol.metadata()
    log.info("loaded %s in %.1fs: %s", pol.policy_type, time.time() - t0, json.dumps(meta))
    if pol.device.startswith("cuda"):
        log.info("GPU memory after load: %.2f GB allocated", torch.cuda.memory_allocated() / 1024**3)
    packer = wire.Packer()
    n_requests = 0

    async def handler(ws):
        nonlocal n_requests
        log.info("client connected from %s", ws.remote_address)
        await ws.send(packer.pack(meta))
        try:
            while True:
                obs = wire.unpackb(await ws.recv())
                n_requests += 1
                try:
                    reply = await asyncio.to_thread(pol.infer, obs)  # keep the event loop free for pings
                except Exception as e:  # report to the client instead of dropping the connection
                    log.exception("request #%d failed", n_requests)
                    reply = {"error": f"{type(e).__name__}: {e}"}
                else:
                    if n_requests <= 3 or n_requests % 100 == 0:
                        t = reply["server_timing"]
                        log.info("request #%d: batch %d -> %d actions in %.0f ms", n_requests, t["batch"],
                                 t["n_action_steps"], t["infer_ms"])
                await ws.send(packer.pack(reply))
        except ConnectionClosed:
            log.info("client disconnected after %d requests", n_requests)

    def health(connection, request):
        if request.path == "/healthz":
            return connection.respond(http.HTTPStatus.OK, "OK\n")
        return None

    async def run():
        async with serve(handler, args.host, args.port, compression=None, max_size=None,
                         process_request=health) as server:
            log.info("LeRobot policy server listening on ws://%s:%d", args.host, args.port)
            await server.serve_forever()

    asyncio.run(run())


if __name__ == "__main__":
    main()
