"""Start OpenPi policy server after runtime SOARM config registration.

This runs registration and serve_policy in the same Python process so
OpenPi's in-memory config registry includes SOARM configs during startup.
"""

from __future__ import annotations

import logging

import tyro

import scripts.serve_policy as serve_policy
from scripts.register_soarm_configs import register


def main() -> None:
    register()
    serve_policy.main(tyro.cli(serve_policy.Args))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main()
