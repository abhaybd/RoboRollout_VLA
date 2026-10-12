from dataclasses import dataclass
from typing import Any, Dict, Iterable, Tuple

import numpy as np


@dataclass(frozen=True)
class Arm:
    """An arm of a robot, as it appears in RoboRollout observations (under qpos) and actions."""

    # key of the arm's joint positions, in radians
    joints: str
    # key of the arm's gripper position, in [0, 1] where 0 is open and 1 is closed
    gripper: str
    n_joints: int


@dataclass(frozen=True)
class Robot:
    """
    The observation and action layout of a RoboRollout robot, for policy servers that support multiple robots.

    Observations have the task, the joint and gripper positions of each arm under qpos, and an (H, W, 3) uint8 image
    for each camera. Actions have absolute joint position targets and gripper positions for each arm, under the
    same keys as in qpos.
    """

    name: str
    arms: Tuple[Arm, ...]

    @property
    def dim(self) -> int:
        """The total number of joints and grippers, i.e. the size of the state and of each action."""
        return sum(arm.n_joints + 1 for arm in self.arms)

    def make_dummy_obs(self, cameras: Iterable[str], task: str = "pick up the object") -> Dict[str, Any]:
        """Create an observation, e.g. for warming up a policy."""
        qpos = {}
        for arm in self.arms:
            qpos[arm.joints] = np.zeros(arm.n_joints)
            qpos[arm.gripper] = np.zeros(1)
        return {
            "task": task,
            "qpos": qpos,
            **{cam: np.zeros((360, 640, 3), dtype=np.uint8) for cam in cameras},
        }
