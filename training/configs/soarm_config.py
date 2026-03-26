"""OpenPi DataConfig for the SO-ARM101 robot arm.

Register at runtime via ``docker/openpi-server/scripts/register_soarm_configs.py``
(see docker compose volume ``training/configs`` -> ``configs_custom``).

Training / LoRA inference use ``soarm_pi0`` / ``soarm_pi0_fast`` (norm stats under
``assets/soarm101`` in your checkpoint). Before fine-tuning, use the ``*_bootstrap``
configs with public OpenPi base checkpoints (norm stats from the base ckpt's Libero assets).
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
import pathlib

from typing_extensions import override

import openpi.models.pi0_config as pi0_config
import openpi.models.pi0_fast as pi0_fast
import openpi.models.model as _model
import openpi.transforms as _transforms
import openpi.training.weight_loaders as weight_loaders
from openpi.training.config import (
    AssetsConfig,
    DataConfig,
    DataConfigFactory,
    ModelTransformFactory,
    TrainConfig,
)


_LIBERO_ASSET_ID = "physical-intelligence/libero"


@dataclasses.dataclass(frozen=True)
class SoarmDataConfig(DataConfigFactory):
    """Maps SO-ARM101 LeRobot keys to OpenPi model inputs."""

    repo_id: str = "local/soarm101_episodes"
    use_delta_joint_actions: bool = False
    default_prompt: str | None = "move the robot arm to the green target"
    assets: AssetsConfig = dataclasses.field(
        default_factory=lambda: AssetsConfig(asset_id="soarm101")
    )
    repack_transforms: _transforms.Group = dataclasses.field(
        default_factory=lambda: _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "images": {
                            "cam_wrist": "observation.images.wrist",
                            "cam_overhead": "observation.images.third_person",
                        },
                        "state": "observation.state",
                        "actions": "action",
                    }
                )
            ]
        )
    )
    action_sequence_keys: Sequence[str] = ("action",)

    @override
    def create(
        self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig
    ) -> DataConfig:
        data_transforms = _transforms.Group(inputs=[], outputs=[])

        if self.use_delta_joint_actions:
            delta_action_mask = _transforms.make_bool_mask(5, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        model_transforms = ModelTransformFactory(
            default_prompt=self.default_prompt
        )(model_config)

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=self.repack_transforms,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            action_sequence_keys=self.action_sequence_keys,
        )


def _soarm_bootstrap_assets_pi0_base() -> AssetsConfig:
    """Norm stats location inside the public ``pi0_base`` checkpoint."""
    return AssetsConfig(
        assets_dir="gs://openpi-assets/checkpoints/pi0_base/assets",
        asset_id=_LIBERO_ASSET_ID,
    )


def _soarm_bootstrap_assets_pi0_fast_base() -> AssetsConfig:
    """Norm stats location inside the public ``pi0_fast_base`` checkpoint."""
    return AssetsConfig(
        assets_dir="gs://openpi-assets/checkpoints/pi0_fast_base/assets",
        asset_id=_LIBERO_ASSET_ID,
    )


# Configs for your checkpoint (assets/soarm101/...) after data collection + fine-tuning.
SOARM_TRAIN_CONFIGS: tuple[TrainConfig, ...] = (
    TrainConfig(
        name="soarm_pi0",
        model=pi0_config.Pi0Config(action_horizon=10),
        data=SoarmDataConfig(),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "gs://openpi-assets/checkpoints/pi0_base/params"
        ),
    ),
    TrainConfig(
        name="soarm_pi0_fast",
        model=pi0_fast.Pi0FASTConfig(
            action_dim=6, action_horizon=10, max_token_len=180
        ),
        data=SoarmDataConfig(),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "gs://openpi-assets/checkpoints/pi0_fast_base/params"
        ),
    ),
    # Same observation schema as SOARM, but load Libero norm stats shipped with OpenPi base ckpts
    # (required for ``policy:checkpoint --policy.dir=gs://.../pi0_base`` before you have LoRA assets).
    TrainConfig(
        name="soarm_pi0_bootstrap",
        model=pi0_config.Pi0Config(action_horizon=10),
        data=dataclasses.replace(
            SoarmDataConfig(), assets=_soarm_bootstrap_assets_pi0_base()
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "gs://openpi-assets/checkpoints/pi0_base/params"
        ),
    ),
    TrainConfig(
        name="soarm_pi0_fast_bootstrap",
        model=pi0_fast.Pi0FASTConfig(
            action_dim=6, action_horizon=10, max_token_len=180
        ),
        data=dataclasses.replace(
            SoarmDataConfig(), assets=_soarm_bootstrap_assets_pi0_fast_base()
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "gs://openpi-assets/checkpoints/pi0_fast_base/params"
        ),
    ),
)

# Back-compat for docs / external snippets that import SOARM_CONFIGS dict.
SOARM_CONFIGS = {cfg.name: cfg for cfg in SOARM_TRAIN_CONFIGS}
