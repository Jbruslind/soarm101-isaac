"""OpenPi DataConfig for the SO-ARM101 robot arm.

Register this config by adding it to OpenPi's _CONFIGS dict in
src/openpi/training/config.py, or by monkey-patching at runtime.

The SO-ARM101 has 6 DOF (5 arm joints + 1 gripper). Dataset keys, policy keys, joint order,
image size (224x224) and action semantics (absolute joint-position targets at 30 Hz) come from
the shared contract, ``isaac_envs/soarm_common/contract.py``, also used by the sim, the data
collector and the ROS 2 bridge.
"""

from __future__ import annotations

import dataclasses
import importlib
import os
import sys
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


def _load_contract():
    """Import ``soarm_common.contract``: from PYTHONPATH, $SOARM_COMMON_DIR, /opt/soarm (docker
    compose mount), a ``soarm_common/`` copy next to this file (deploy scripts sync one) or the
    repo's isaac_envs/ in a source checkout."""
    try:
        return importlib.import_module("soarm_common.contract")
    except ImportError:
        pass
    here = pathlib.Path(os.path.realpath(__file__)).parent
    for cand in (os.environ.get("SOARM_COMMON_DIR"), "/opt/soarm", str(here), str(here.parents[1] / "isaac_envs")):
        if cand and (pathlib.Path(cand) / "soarm_common" / "contract.py").is_file():
            sys.path.insert(0, cand)
            return importlib.import_module("soarm_common.contract")
    raise ImportError("soarm_common.contract not found; mount isaac_envs/soarm_common or set SOARM_COMMON_DIR")


C = _load_contract()


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
                        C.POLICY_IMAGES_KEY: {cam.policy_key: cam.dataset_key for cam in C.CAMERAS},
                        C.POLICY_STATE_KEY: C.DATASET_STATE_KEY,
                        C.POLICY_ACTIONS_KEY: C.DATASET_ACTION_KEY,
                    }
                )
            ]
        )
    )

    action_sequence_keys: Sequence[str] = (C.DATASET_ACTION_KEY,)

    @override
    def create(
        self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig
    ) -> DataConfig:
        data_transforms = _transforms.Group(
            inputs=[],
            outputs=[],
        )

        if self.use_delta_joint_actions:
            # arm joints delta, gripper absolute
            delta_action_mask = _transforms.make_bool_mask(C.NUM_ARM_JOINTS, -(C.NUM_JOINTS - C.NUM_ARM_JOINTS))
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
