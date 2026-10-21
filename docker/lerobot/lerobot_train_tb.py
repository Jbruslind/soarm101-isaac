"""`lerobot-train` with TensorBoard logging instead of Weights & Biases.

lerobot 0.6.1 only ships a W&B logger. This launcher swaps it for a small
TensorBoard writer with the same interface and then runs the stock trainer, so
every `lerobot-train` flag works unchanged. Pass `--wandb.enable=true` to turn
logging on (scripts/train_lerobot.sh does this); scalars land in
<output_dir>/tensorboard. Set SOARM_LOGGER=wandb to keep the real W&B logger.

    python /opt/soarm/lerobot_train_tb.py --policy.type=act --dataset.repo_id=... --wandb.enable=true
"""

import numbers
import os
from pathlib import Path


class TensorBoardLogger:
    """Drop-in for lerobot.common.wandb_utils.WandBLogger (the calls lerobot_train makes)."""

    def __init__(self, cfg):
        from torch.utils.tensorboard import SummaryWriter

        log_dir = Path(cfg.output_dir) / "tensorboard"
        self._writer = SummaryWriter(log_dir=str(log_dir))
        print(f"TensorBoard logs: {log_dir}", flush=True)

    def log_dict(self, d, step=None, mode="train", custom_step_key=None):
        if step is None and custom_step_key is not None:
            step = d.get(custom_step_key)
        for key, value in d.items():
            if isinstance(value, bool) or not isinstance(value, numbers.Number):
                continue
            self._writer.add_scalar(f"{mode}/{key}", value, step)
        self._writer.flush()

    def log_policy(self, checkpoint_dir):
        pass

    def log_video(self, video_path, step, mode="train"):
        pass


def main():
    from lerobot.scripts import lerobot_train

    if os.environ.get("SOARM_LOGGER", "tensorboard") == "tensorboard":
        lerobot_train.WandBLogger = TensorBoardLogger
    lerobot_train.main()


if __name__ == "__main__":
    main()
