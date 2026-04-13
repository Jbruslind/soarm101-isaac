"""OpenPi DataConfig for the SO-ARM101 robot arm.

Register this config by adding it to OpenPi's _CONFIGS dict in
src/openpi/training/config.py, or by monkey-patching at runtime.

The SO-ARM101 has 6 DOF (5 arm joints + 1 gripper).  Observations include:
  - 1-2 camera images (wrist, optionally overhead) at 224x224
  - 6-element state vector (joint positions)

Actions are 6-element vectors (absolute joint position targets).
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
import pathlib

from typing_extensions import override

import openpi.models.model as _model
import openpi.transforms as _transforms
from openpi.training.config import (
    DataConfig,
    DataConfigFactory,
    ModelTransformFactory,
    TrainConfig,
    AssetsConfig,
)


@dataclasses.dataclass(frozen=True)
class SoarmDataConfig(DataConfigFactory):
    """Data configuration for the SO-ARM101 robot arm.

    Maps LeRobot dataset keys to the format expected by OpenPi models.
    Follows the pattern established by LeRobotAlohaDataConfig and
    LeRobotLiberoDataConfig in the upstream OpenPi codebase.
    """

    repo_id: str = "local/soarm101_episodes"

    use_delta_joint_actions: bool = False

    default_prompt: str | None = "move the robot arm to the green target"

    assets: AssetsConfig = dataclasses.field(
        default_factory=lambda: AssetsConfig(asset_id="soarm101")
    )

    repack_transforms: _transforms.Group = dataclasses.field(
        default=_transforms.Group(
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
        data_transforms = _transforms.Group(
            inputs=[],
            outputs=[],
        )

        if self.use_delta_joint_actions:
            delta_action_mask = _transforms.make_bool_mask(5, -1)  # 5 arm joints delta, gripper absolute
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


# Config entries to register with OpenPi's config system.
# Add these to _CONFIGS in openpi/training/config.py or load dynamically.
SOARM_CONFIGS = {
    "soarm_pi0": TrainConfig(
        model_type=_model.ModelType.PI0,
        data=SoarmDataConfig(),
    ),
    "soarm_pi0_fast": TrainConfig(
        model_type=_model.ModelType.PI0_FAST,
        data=SoarmDataConfig(),
    ),
}
