"""
Serve a MolmoBot policy (e.g. allenai/MolmoBot-DROID) to RoboRollout's DROID robot.
Requires the `molmobot` extra.
"""

import logging
from types import SimpleNamespace
from typing import Any, Dict

import numpy as np

from robo_rollout.policy.policy_server import InferencePolicy

from roborollout_vla import droid
from roborollout_vla.checkpoints import resolve_checkpoint
from roborollout_vla.server import make_parser, serve, warmup

# camera names expected by MolmoBot's RealRobotVLAPolicyConfig
MOLMOBOT_EXO_CAMERA = "exo_camera_1"
MOLMOBOT_WRIST_CAMERA = "wrist_camera"


class MolmoBotPolicy(InferencePolicy):
    """
    Wraps MolmoBot's RealRobotVLAPolicy, which buffers action chunks itself,
    converting between RoboRollout's and MolmoSpaces' gripper conventions.
    """

    def __init__(self, policy, exo_camera: str = droid.EXO_CAMERA, wrist_camera: str = droid.WRIST_CAMERA):
        """
        Args:
            policy: The MolmoBot policy (olmo.eval.configure_real_robot.RealRobotVLAPolicy).
            exo_camera: The observation key of the exterior camera image.
            wrist_camera: The observation key of the wrist camera image.
        """
        self._policy = policy
        self.exo_camera = exo_camera
        self.wrist_camera = wrist_camera

    def reset(self) -> None:
        self._policy.reset()

    def get_action(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        action = self._policy.get_action({
            "task": obs["task"],
            "qpos": {
                "arm": np.asarray(obs["qpos"]["arm"]),
                "gripper": droid.to_molmospaces_gripper(obs["qpos"]["gripper"]),
            },
            MOLMOBOT_EXO_CAMERA: obs[self.exo_camera],
            MOLMOBOT_WRIST_CAMERA: obs[self.wrist_camera],
        })
        return {
            "arm": np.array(action["arm"][:7]),
            # MolmoBot predicts the gripper in [0, 255]
            "gripper": np.asarray(action["gripper"], dtype=float).reshape(1) / 255.0,
        }


def main():
    parser = make_parser("Serve a MolmoBot policy")
    parser.add_argument(
        "--checkpoint", default="allenai/MolmoBot-DROID", help="Local checkpoint directory or HuggingFace repo id"
    )
    parser.add_argument(
        "--action-type",
        choices=["joint_pos", "joint_pos_rel"],
        default="joint_pos",
        help="Arm action type the policy was trained with",
    )
    parser.add_argument("--exo-camera", default=droid.EXO_CAMERA, help="Observation key of the exterior camera")
    parser.add_argument("--wrist-camera", default=droid.WRIST_CAMERA, help="Observation key of the wrist camera")
    args = parser.parse_args()

    from olmo.eval.configure_real_robot import RealRobotVLAPolicy, RealRobotVLAPolicyConfig

    logging.basicConfig(level=logging.INFO, force=True)

    checkpoint_dir = resolve_checkpoint(args.checkpoint)
    policy_config = RealRobotVLAPolicyConfig(
        checkpoint_path=checkpoint_dir,
        camera_names=[MOLMOBOT_EXO_CAMERA, MOLMOBOT_WRIST_CAMERA],
        action_type=args.action_type,
        action_keys={"arm": args.action_type, "gripper": "joint_pos"},
    )
    # RealRobotVLAPolicy only reads the policy config from the experiment config
    molmobot_policy = RealRobotVLAPolicy(config=SimpleNamespace(policy_config=policy_config), task_type="manipulation")
    policy = MolmoBotPolicy(molmobot_policy, exo_camera=args.exo_camera, wrist_camera=args.wrist_camera)
    warmup(policy, droid.make_dummy_obs([args.exo_camera, args.wrist_camera]))
    serve(policy, args.checkpoint, args, metadata={"policy_type": "molmobot", "checkpoint_dir": checkpoint_dir})


if __name__ == "__main__":
    main()
