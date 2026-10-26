"""msgpack-numpy encoding of the policy-server wire protocol (OpenPi ``WebsocketPolicyServer``).

Byte-compatible with ``openpi_client.msgpack_numpy``, which the ros2-bridge and the mock server
use, so any of the three can talk to any other. It is copied here so the LeRobot policy server
and the Isaac Sim client need only ``msgpack`` (not the whole ``openpi-client`` package).

Arrays travel as ``{b"__ndarray__": True, b"data": raw bytes, b"dtype": dtype.str, b"shape": shape}``;
numpy scalars as ``{b"__npgeneric__": True, b"data": value, b"dtype": dtype.str}``. Object and void
arrays are refused (no pickle on the wire).

Protocol, as served by ``training/scripts/lerobot_policy_server.py``:

* on connect the server sends one packed metadata dict;
* each request is a packed observation dict ``{"state", "images": {policy_key: HxWx3 uint8},
  "prompt"}`` (keys from :mod:`soarm_common.contract`), optionally batched (state ``(N, 6)``,
  images ``(N, H, W, 3)``, prompt a list) and optionally with ``"n_action_steps"``;
* each reply is ``{"actions": float32 (H, 6) or (N, H, 6), "server_timing": {...}}`` or
  ``{"error": "..."}``;
* ``GET /healthz`` returns 200.
"""

from __future__ import annotations

import functools

import msgpack
import numpy as np


def pack_array(obj):
    if isinstance(obj, (np.ndarray, np.generic)) and obj.dtype.kind in ("V", "O", "c"):
        raise ValueError(f"Unsupported dtype: {obj.dtype}")
    if isinstance(obj, np.ndarray):
        return {b"__ndarray__": True, b"data": obj.tobytes(), b"dtype": obj.dtype.str, b"shape": obj.shape}
    if isinstance(obj, np.generic):
        return {b"__npgeneric__": True, b"data": obj.item(), b"dtype": obj.dtype.str}
    return obj


def unpack_array(obj):
    if b"__ndarray__" in obj:
        return np.ndarray(buffer=obj[b"data"], dtype=np.dtype(obj[b"dtype"]), shape=obj[b"shape"])
    if b"__npgeneric__" in obj:
        return np.dtype(obj[b"dtype"]).type(obj[b"data"])
    return obj


Packer = functools.partial(msgpack.Packer, default=pack_array)
packb = functools.partial(msgpack.packb, default=pack_array)
unpackb = functools.partial(msgpack.unpackb, object_hook=unpack_array)
