"""List the registered Soarm tasks with Isaac Lab's ``list_envs`` table (no simulator needed).

    /isaac-sim/python.sh /isaac_envs/soarm_tasks/scripts/list_tasks.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import soarm_tasks  # noqa: E402,F401  (registers the tasks)
from isaaclab.cli.commands.list_envs import command_list_envs  # noqa: E402

if __name__ == "__main__":
    command_list_envs(["--keyword", "Soarm", *sys.argv[1:]])
