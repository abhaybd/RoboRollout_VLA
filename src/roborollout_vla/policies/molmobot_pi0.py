"""
Serve a MolmoBot-Pi0 policy (e.g. allenai/MolmoBot-Pi0-DROID) to RoboRollout's DROID robot.
Requires the `molmobot-pi0` extra.

This replicates molmobot_pi0.eval.policies.pi rather than wrapping it, since that module depends on MolmoSpaces,
whose dependencies conflict with openpi's.
"""

import dataclasses
import logging
import pickle
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from robo_rollout.utils.image import resize_with_crop, resize_with_pad

from roborollout_vla.checkpoints import resolve_checkpoint
from roborollout_vla.policies.chunking import ChunkedPolicy
from roborollout_vla.policies.molmospaces import to_molmospaces_gripper
from roborollout_vla.robots import droid
from roborollout_vla.server import make_parser, serve, warmup


class MolmoBotPi0Policy(ChunkedPolicy):
    """
    Wraps an openpi policy trained with MolmoBot-Pi0, whose actions are 7 joint positions or velocities
    followed by the gripper position.
    """

    def __init__(
        self,
        policy,
        joint_velocity: bool,
        open_loop_horizon: Optional[int] = None,
        exo_camera: str = droid.EXO_CAMERA,
        wrist_camera: str = droid.WRIST_CAMERA,
    ):
        """
        Args:
            policy: The openpi policy (openpi.policies.policy.Policy).
            joint_velocity: Whether the policy predicts joint velocities, instead of joint positions.
            open_loop_horizon: How many actions of each predicted chunk to execute. If None, executes the whole chunk.
            exo_camera: The observation key of the exterior camera image.
            wrist_camera: The observation key of the wrist camera image.
        """
        super().__init__(open_loop_horizon)
        self._policy = policy
        self.joint_velocity = joint_velocity
        self.exo_camera = exo_camera
        self.wrist_camera = wrist_camera

    @staticmethod
    def _preprocess_image(image: np.ndarray) -> np.ndarray:
        # match the 360p training data, then resize to the model input size
        return resize_with_pad(resize_with_crop(image, 360, 640), 224, 224)

    def predict_chunk(self, obs: Dict[str, Any]) -> np.ndarray:
        return self._policy.infer({
            "observation/exterior_image_1_left": self._preprocess_image(obs[self.exo_camera]),
            "observation/wrist_image_left": self._preprocess_image(obs[self.wrist_camera]),
            "observation/joint_position": np.asarray(obs["qpos"]["arm"]),
            # the policy's input transforms expect the MolmoSpaces gripper representation
            "observation/gripper_position": to_molmospaces_gripper(obs["qpos"]["gripper"]),
            "prompt": obs["task"],
        })["actions"]

    def to_action(self, obs: Dict[str, Any], action: np.ndarray) -> Dict[str, Any]:
        if self.joint_velocity:
            arm = droid.joint_velocity_to_position(obs["qpos"]["arm"], action[:7])
        else:
            arm = np.array(action[:7])
        return {
            "arm": arm,
            "gripper": np.clip(action[7:8], 0.0, 1.0),
        }


def main():
    parser = make_parser("Serve a MolmoBot-Pi0 policy")
    parser.add_argument(
        "--checkpoint", default="allenai/MolmoBot-Pi0-DROID", help="Local checkpoint directory or HuggingFace repo id"
    )
    parser.add_argument(
        "--open-loop-horizon", type=int, help="Actions to execute from each chunk. Executes the whole chunk if unset."
    )
    parser.add_argument(
        "--compile-mode",
        default="max-autotune-no-cudagraphs",
        help="torch.compile mode for PyTorch checkpoints",
    )
    parser.add_argument("--device", help="Torch device for PyTorch checkpoints. Defaults to cuda if available.")
    parser.add_argument("--exo-camera", default=droid.EXO_CAMERA, help="Observation key of the exterior camera")
    parser.add_argument("--wrist-camera", default=droid.WRIST_CAMERA, help="Observation key of the wrist camera")
    args = parser.parse_args()

    import torch
    from openpi.policies import policy_config

    import molmobot_pi0.config_openpi  # noqa: F401 (registers the configs that train configs are pickled with)

    logging.basicConfig(level=logging.INFO, force=True)
    # disable cudagraphs to avoid threading issues
    torch._inductor.config.triton.cudagraphs = False

    checkpoint_dir = Path(resolve_checkpoint(args.checkpoint))
    with open(checkpoint_dir / "assets" / "train_config.pkl", "rb") as f:
        train_config = pickle.load(f)
    train_config = dataclasses.replace(
        train_config,
        model=dataclasses.replace(train_config.model, pytorch_compile_mode=args.compile_mode),
    )
    joint_velocity = not train_config.data.joint_pos_actions

    openpi_policy = policy_config.create_trained_policy(train_config, checkpoint_dir, pytorch_device=args.device)
    policy = MolmoBotPi0Policy(
        openpi_policy,
        joint_velocity=joint_velocity,
        open_loop_horizon=args.open_loop_horizon,
        exo_camera=args.exo_camera,
        wrist_camera=args.wrist_camera,
    )
    warmup(policy, droid.make_dummy_obs([args.exo_camera, args.wrist_camera]))
    serve(policy, train_config.name, args, metadata={
        "policy_type": "molmobot-pi0",
        "checkpoint_dir": str(checkpoint_dir),
        "action_space": "joint_velocity" if joint_velocity else "joint_position",
    })


if __name__ == "__main__":
    main()
