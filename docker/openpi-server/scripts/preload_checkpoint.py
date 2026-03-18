# Preload default DROID checkpoint so the first inference isn't delayed.
# Run from /app/openpi with: uv run --python 3.11 scripts/preload_checkpoint.py

from openpi.policies import policy_config
from openpi.training import config

policy_config.create_trained_policy(
    config.get_config("pi05_droid"),
    "gs://openpi-assets/checkpoints/pi05_droid",
)
print("Preload done.")
