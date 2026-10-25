"""Locate and import the shared SO-ARM101 contract (``isaac_envs/soarm_common/contract.py``).

The contract is the single source of the joint names, camera keys/topics, image size and
control rate shared with the Isaac Sim side. It is plain Python. Search order:

1. ``soarm_common`` already importable (e.g. on ``PYTHONPATH``);
2. ``$SOARM_COMMON_DIR`` (directory that contains ``soarm_common/``);
3. ``/opt/soarm`` (where docker/docker-compose.yml mounts it in the ros2-bridge container);
4. ``<repo>/isaac_envs`` relative to this file in a source checkout.
"""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path


def _load():
    try:
        return importlib.import_module("soarm_common.contract")
    except ImportError:
        pass
    here = Path(os.path.realpath(__file__))
    candidates = [os.environ.get("SOARM_COMMON_DIR"), "/opt/soarm"]
    if len(here.parents) > 4:
        candidates.append(str(here.parents[4] / "isaac_envs"))
    for cand in candidates:
        if cand and (Path(cand) / "soarm_common" / "contract.py").is_file():
            sys.path.insert(0, cand)
            return importlib.import_module("soarm_common.contract")
    raise ImportError(
        "soarm_common.contract not found. Mount isaac_envs/soarm_common into the container "
        "(docker-compose.yml mounts it at /opt/soarm/soarm_common) or set SOARM_COMMON_DIR to the "
        "directory that contains soarm_common/."
    )


contract = _load()
