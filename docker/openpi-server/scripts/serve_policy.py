# Python 3.8–compatible version of OpenPi scripts/serve_policy.py for Jetson.
# Upstream uses match/case (3.10+), str|None, and typing.TypeAlias (3.10+).
# This file and the shim below allow L4T (Python 3.8) to run the server.

import sys

# TypeAlias exists in typing only from Python 3.10; inject from typing_extensions for 3.8/3.9
if sys.version_info < (3, 10):
    import typing
    import typing_extensions
    typing.TypeAlias = typing_extensions.TypeAlias

import dataclasses
import enum
import logging
import socket
from typing import Dict, Optional, Union

import tyro

from openpi.policies import policy as _policy
from openpi.policies import policy_config as _policy_config
from openpi.serving import websocket_policy_server
from openpi.training import config as _config


class EnvMode(enum.Enum):
    """Supported environments."""

    ALOHA = "aloha"
    ALOHA_SIM = "aloha_sim"
    DROID = "droid"
    LIBERO = "libero"


@dataclasses.dataclass
class Checkpoint:
    """Load a policy from a trained checkpoint."""

    config: str
    dir: str


@dataclasses.dataclass
class Default:
    """Use the default policy for the given environment."""


@dataclasses.dataclass
class Args:
    """Arguments for the serve_policy script."""

    env: EnvMode = EnvMode.ALOHA_SIM
    default_prompt: Optional[str] = None
    port: int = 8000
    record: bool = False
    policy: Union[Checkpoint, Default] = dataclasses.field(default_factory=Default)


DEFAULT_CHECKPOINT: Dict[EnvMode, Checkpoint] = {
    EnvMode.ALOHA: Checkpoint(
        config="pi05_aloha",
        dir="gs://openpi-assets/checkpoints/pi05_base",
    ),
    EnvMode.ALOHA_SIM: Checkpoint(
        config="pi0_aloha_sim",
        dir="gs://openpi-assets/checkpoints/pi0_aloha_sim",
    ),
    EnvMode.DROID: Checkpoint(
        config="pi05_droid",
        dir="gs://openpi-assets/checkpoints/pi05_droid",
    ),
    EnvMode.LIBERO: Checkpoint(
        config="pi05_libero",
        dir="gs://openpi-assets/checkpoints/pi05_libero",
    ),
}


def create_default_policy(
    env: EnvMode, *, default_prompt: Optional[str] = None
) -> _policy.Policy:
    """Create a default policy for the given environment."""
    checkpoint = DEFAULT_CHECKPOINT.get(env)
    if checkpoint:
        return _policy_config.create_trained_policy(
            _config.get_config(checkpoint.config), checkpoint.dir, default_prompt=default_prompt
        )
    raise ValueError(f"Unsupported environment mode: {env}")


def create_policy(args: Args) -> _policy.Policy:
    """Create a policy from the given arguments (Python 3.9 compatible: no match/case)."""
    if isinstance(args.policy, Checkpoint):
        return _policy_config.create_trained_policy(
            _config.get_config(args.policy.config),
            args.policy.dir,
            default_prompt=args.default_prompt,
        )
    if isinstance(args.policy, Default):
        return create_default_policy(args.env, default_prompt=args.default_prompt)
    raise ValueError(f"Unexpected policy type: {type(args.policy)}")


def main(args: Args) -> None:
    policy = create_policy(args)
    policy_metadata = policy.metadata

    if args.record:
        policy = _policy.PolicyRecorder(policy, "policy_records")

    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logging.info("Creating server (host: %s, ip: %s)", hostname, local_ip)

    server = websocket_policy_server.WebsocketPolicyServer(
        policy=policy,
        host="0.0.0.0",
        port=args.port,
        metadata=policy_metadata,
    )
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
