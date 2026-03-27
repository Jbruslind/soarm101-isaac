"""Start OpenPi policy server after runtime SOARM config registration.

This runs registration and serve_policy in the same Python process so
OpenPi's in-memory config registry includes SOARM configs during startup.
"""

from __future__ import annotations

import logging

import tyro

import serve_policy
from register_soarm_configs import register


def _install_norm_stats_fallback() -> None:
    # Base OpenPi checkpoints do not always use the same asset_id directory name.
    # Try the requested asset first, then common alternatives.
    from openpi.training import checkpoints as _checkpoints

    original = _checkpoints.load_norm_stats

    def _load_with_fallback(assets_dir, asset_id):
        tried = []
        candidates = [asset_id, "libero", "droid", "trossen", "physical-intelligence/libero"]
        seen = set()
        for candidate in candidates:
            if not candidate or candidate in seen:
                continue
            seen.add(candidate)
            tried.append(candidate)
            try:
                return original(assets_dir, candidate)
            except FileNotFoundError:
                continue
        raise FileNotFoundError(
            f"Norm stats file not found under {assets_dir}. Tried asset IDs: {', '.join(tried)}"
        )

    _checkpoints.load_norm_stats = _load_with_fallback


def main() -> None:
    register()
    _install_norm_stats_fallback()
    serve_policy.main(tyro.cli(serve_policy.Args))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main()
