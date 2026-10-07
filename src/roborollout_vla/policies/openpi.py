"""
Serve an openpi DROID policy (e.g. pi0_droid, pi0_fast_droid, pi05_droid) to RoboRollout's DROID robot.
Requires the `openpi` extra.
"""

import logging
from typing import Any, Dict

import numpy as np

from roborollout_vla.policies.chunking import ChunkedPolicy
from roborollout_vla.robots import droid
from roborollout_vla.server import make_parser, serve, warmup

JOINT_VELOCITY = "joint_velocity"
JOINT_POSITION = "joint_position"


class OpenPIPolicy(ChunkedPolicy):
    """
    Wraps an openpi policy that uses openpi's DROID inputs/outputs, with actions of 7 joint velocities or positions
    followed by the gripper position. Converts actions the same way as openpi's DROID example client.
    """

    def __init__(
        self,
        policy,
        action_space: str = JOINT_VELOCITY,
        open_loop_horizon: int = 8,
        exo_camera: str = droid.EXO_CAMERA,
        wrist_camera: str = droid.WRIST_CAMERA,
    ):
        """
        Args:
            policy: The openpi policy (openpi.policies.policy.Policy).
            action_space: The policy's arm action space, either "joint_velocity" or "joint_position".
            open_loop_horizon: How many actions of each predicted chunk to execute.
            exo_camera: The observation key of the exterior camera image.
            wrist_camera: The observation key of the wrist camera image.
        """
        assert action_space in (JOINT_VELOCITY, JOINT_POSITION), f"Invalid action space: {action_space}"
        super().__init__(open_loop_horizon)
        self._policy = policy
        self.action_space = action_space
        self.exo_camera = exo_camera
        self.wrist_camera = wrist_camera

    def predict_chunk(self, obs: Dict[str, Any]) -> np.ndarray:
        # images are resized by the policy's input transforms
        return self._policy.infer({
            "observation/exterior_image_1_left": obs[self.exo_camera],
            "observation/wrist_image_left": obs[self.wrist_camera],
            "observation/joint_position": np.asarray(obs["qpos"]["arm"]),
            "observation/gripper_position": np.asarray(obs["qpos"]["gripper"]),
            "prompt": obs["task"],
        })["actions"]

    def to_action(self, obs: Dict[str, Any], action: np.ndarray) -> Dict[str, Any]:
        if self.action_space == JOINT_VELOCITY:
            arm = droid.joint_velocity_to_position(obs["qpos"]["arm"], np.clip(action[:7], -1.0, 1.0))
        else:
            arm = np.array(action[:7])
        return {
            "arm": arm,
            "gripper": np.array([1.0 if action[7] > 0.5 else 0.0]),
        }


def main():
    parser = make_parser("Serve an openpi DROID policy")
    parser.add_argument("--config", default="pi05_droid", help="Name of the openpi training config")
    parser.add_argument(
        "--checkpoint",
        help="Checkpoint directory, local or remote (e.g. gs://...). Defaults to gs://openpi-assets/checkpoints/<config>",
    )
    parser.add_argument(
        "--action-space",
        choices=[JOINT_VELOCITY, JOINT_POSITION],
        default=JOINT_VELOCITY,
        help="Arm action space of the policy. The official DROID policies use joint velocity.",
    )
    parser.add_argument("--open-loop-horizon", type=int, default=8, help="Actions to execute from each chunk")
    parser.add_argument("--exo-camera", default=droid.EXO_CAMERA, help="Observation key of the exterior camera")
    parser.add_argument("--wrist-camera", default=droid.WRIST_CAMERA, help="Observation key of the wrist camera")
    args = parser.parse_args()
    if args.checkpoint is None:
        args.checkpoint = f"gs://openpi-assets/checkpoints/{args.config}"

    from openpi.policies import policy_config
    from openpi.training import config as _config

    logging.basicConfig(level=logging.INFO, force=True)

    openpi_policy = policy_config.create_trained_policy(_config.get_config(args.config), args.checkpoint)
    policy = OpenPIPolicy(
        openpi_policy,
        action_space=args.action_space,
        open_loop_horizon=args.open_loop_horizon,
        exo_camera=args.exo_camera,
        wrist_camera=args.wrist_camera,
    )
    warmup(policy, droid.make_dummy_obs([args.exo_camera, args.wrist_camera]))
    serve(policy, args.config, args, metadata={"policy_type": "openpi", **openpi_policy.metadata})


if __name__ == "__main__":
    main()
