"""
MolmoSpaces conventions, which the MolmoBot policies are trained with.
"""

import numpy as np

# MolmoSpaces represents the Franka gripper as two finger joint positions in [0, GRIPPER_CLOSED], where 0 is open.
GRIPPER_CLOSED = 0.824033


def to_molmospaces_gripper(gripper: np.ndarray) -> np.ndarray:
    """Convert a RoboRollout gripper position, in [0, 1] where 0 is open, to the MolmoSpaces gripper qpos."""
    return np.full(2, np.asarray(gripper).item() * GRIPPER_CLOSED)
