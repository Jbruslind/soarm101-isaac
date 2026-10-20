"""Shared SO-ARM101 robot / camera / control contract.

``soarm_common.contract`` is plain Python (stdlib only). It is the single source of the joint
names, camera poses and keys, control rate and action semantics, and is imported outside Isaac
Sim too: by the ROS 2 bridge, the OpenPi training config and the mock policy server.

``soarm_common.robot``, ``soarm_common.cameras`` and ``soarm_common.scene`` build Isaac Lab
configs from it and need Isaac Lab. This ``__init__`` imports none of them, so
``import soarm_common.contract`` works without Isaac Lab or torch.
"""
