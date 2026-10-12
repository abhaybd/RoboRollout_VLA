"""
Observation and action conventions of RoboRollout's bimanual i2rt YAM robot (scripts/yam/run_policy.py in RoboRollout).

Observations:
    task: the language instruction
    qpos/left_arm, qpos/right_arm: (6,) joint angles in radians
    qpos/left_gripper, qpos/right_gripper: (1,) in [0, 1], where 0 is open and 1 is closed
    <camera name>: (H, W, 3) uint8 image, for each camera in the rollout config's policy_cameras

Actions:
    left_arm, right_arm: (6,) absolute joint position targets in radians
    left_gripper, right_gripper: (1,) in [0, 1], where 0 is open and 1 is closed

Note that i2rt itself (and datasets recorded with it, like MolmoAct2's) normalizes the gripper so that 1 is open.
"""

from roborollout_vla.robots.base import Arm, Robot

# default policy camera names in RoboRollout's YAM config
TOP_CAMERA = "top_camera"
LEFT_WRIST_CAMERA = "left_wrist_camera"
RIGHT_WRIST_CAMERA = "right_wrist_camera"

ROBOT = Robot("yam", (Arm("left_arm", "left_gripper", 6), Arm("right_arm", "right_gripper", 6)))
make_dummy_obs = ROBOT.make_dummy_obs
