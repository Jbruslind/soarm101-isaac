"""Run an Isaac Lab script with the Soarm tasks registered.

Isaac Lab's scripts import ``isaaclab_tasks`` only. This wrapper puts ``isaac_envs`` on
``sys.path``, imports :mod:`soarm_tasks` (which registers the gym ids) and then runs the given
script as ``__main__`` with the remaining arguments. Usage, in the isaac-sim container::

    /isaac-sim/python.sh /isaac_envs/soarm_tasks/scripts/with_soarm_tasks.py \
        /workspace/isaaclab/scripts/environments/list_envs.py --keyword Soarm
    /isaac-sim/python.sh /isaac_envs/soarm_tasks/scripts/with_soarm_tasks.py \
        /workspace/isaaclab/scripts/tools/record_demos.py --task Soarm-PickPlace-Fastener-IK-Abs-v0 ...
"""

import os
import runpy
import sys

ISAAC_ENVS = os.environ.get("SOARM_ISAAC_ENVS", os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    sys.path.insert(0, ISAAC_ENVS)
    import soarm_tasks  # noqa: F401  (registers the tasks)

    script = sys.argv[1]
    sys.argv = sys.argv[1:]
    sys.path.insert(0, os.path.dirname(os.path.abspath(script)))
    runpy.run_path(script, run_name="__main__")
