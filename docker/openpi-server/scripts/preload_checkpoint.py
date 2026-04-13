# Preload default DROID checkpoint so the first inference isn't delayed.
# Run from /app/openpi with: uv run --python 3.11 scripts/preload_checkpoint.py

from openpi.policies import policy_config
from openpi.training import config

# Preload both DROID checkpoints:
# - pi05_droid: original DROID fallback (pi05 model; can be slow on Jetson)
# - pi0_fast_droid: faster DROID variant used by our Jetson fallback
policy_config.create_trained_policy(
    config.get_config("pi05_droid"),
    "gs://openpi-assets/checkpoints/pi05_droid",
)
policy_config.create_trained_policy(
    config.get_config("pi0_fast_droid"),
    "gs://openpi-assets/checkpoints/pi0_fast_droid",
)

print("Preload done (pi05_droid + pi0_fast_droid).")
