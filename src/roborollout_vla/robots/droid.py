"""
Observation and action conventions of RoboRollout's DROID robot (scripts/droid/run_policy.py in RoboRollout).

Observations:
    task: the language instruction
    qpos/arm: (7,) joint angles in radians
    qpos/gripper: (1,) in [0, 1], where 0 is open and 1 is closed
    <camera name>: (H, W, 3) uint8 image, for each camera in the rollout config's policy_cameras

Actions:
    arm: (7,) absolute joint position targets in radians
    gripper: (1,) in [0, 1], where 0 is open and 1 is closed
"""

import numpy as np

from roborollout_vla.robots.base import Arm, Robot

# default policy camera names in RoboRollout's DROID config
EXO_CAMERA = "exo_camera_1"
WRIST_CAMERA = "wrist_camera"

# DROID's limit on joint motion per step, in radians. Joint velocity actions are joint deltas normalized by this.
MAX_JOINT_DELTA = 0.2


def joint_velocity_to_position(qpos: np.ndarray, velocity: np.ndarray, max_joint_delta: float = MAX_JOINT_DELTA) -> np.ndarray:
    """
    Convert a DROID joint velocity action to an absolute joint position target.
    Like DROID, if any velocity exceeds 1 then all velocities are scaled down so that the largest is 1.
    """
    velocity = np.asarray(velocity)
    velocity = velocity / max(1.0, np.abs(velocity).max())
    return np.asarray(qpos) + velocity * max_joint_delta


ROBOT = Robot("droid", (Arm("arm", "gripper", 7),))
make_dummy_obs = ROBOT.make_dummy_obs
