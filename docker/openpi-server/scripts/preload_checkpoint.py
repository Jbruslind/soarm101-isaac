# Preload checkpoints so the first inference isn't delayed by large downloads.
# Run from /app/openpi with: uv run --python 3.11 scripts/preload_checkpoint.py
#
# Honors OPENPI_POLICY_MODE:
#   - droid (default): warm pi05_droid via create_trained_policy (legacy Jetson behavior)
#   - soarm: download public base weights only (pi0_fast_base / pi0_base) — no LoRA required

from __future__ import annotations

import os

from openpi.shared import download as dl


def main() -> None:
    mode = os.environ.get("OPENPI_POLICY_MODE", "droid")

    if mode == "soarm":
        cfg = os.environ.get("OPENPI_POLICY_CONFIG", "soarm_pi0_fast")
        if cfg == "soarm_pi0":
            url = "gs://openpi-assets/checkpoints/pi0_base"
        else:
            # soarm_pi0_fast and other pi0-FAST SOARM configs
            url = "gs://openpi-assets/checkpoints/pi0_fast_base"
        dl.maybe_download(url)
        print(f"Preload SOARM base checkpoint done ({url}).")
        return

    from openpi.policies import policy_config
    from openpi.training import config

    policy_config.create_trained_policy(
        config.get_config("pi05_droid"),
        "gs://openpi-assets/checkpoints/pi05_droid",
    )
    print("Preload pi05_droid done.")


if __name__ == "__main__":
    main()
