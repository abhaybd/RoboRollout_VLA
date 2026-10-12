"""
Serve a LeRobot policy of any type that lerobot supports (e.g. act, diffusion, pi0, pi05, smolvla, groot, xvla).
Requires the `lerobot` extra, which requires python 3.12+.
"""

from contextlib import nullcontext
import logging
from typing import Any, Dict, Optional

import numpy as np
import torch

from robo_rollout.utils.image import resize_with_pad

from roborollout_vla.policies.chunking import ChunkedPolicy
from roborollout_vla.robots import ROBOTS, Robot, droid
from roborollout_vla.server import make_parser, serve, warmup

logger = logging.getLogger(__name__)

JOINT_POSITION = "joint_position"
JOINT_VELOCITY = "joint_velocity"
# joint position offsets from the joint positions when the chunk was predicted
JOINT_CHUNK_DELTA = "joint_chunk_delta"
ACTION_SPACES = (JOINT_POSITION, JOINT_VELOCITY, JOINT_CHUNK_DELTA)

IMAGE_PREFIX = "observation.images."
STATE_KEY = "observation.state"


def resolve_image_feature(name: str, image_features: Dict[str, Any]) -> str:
    """Get the key of an image feature, given either the full key or the camera name after "observation.images."."""
    for key in (name, IMAGE_PREFIX + name):
        if key in image_features:
            return key
    raise ValueError(f"Policy has no image feature {name!r}. Its image features are: {list(image_features)}")


class LeRobotPolicy(ChunkedPolicy):
    """
    Wraps a lerobot policy and its pre- and postprocessors, for any robot in roborollout_vla.robots.

    The policy's observation.state is the joint positions of each arm of the robot followed by its gripper position
    (e.g. 7 + 1 for DROID, and 6 + 1 for the left and then the right arm of the YAM), zero-padded if the policy
    expects a larger state. Its actions are laid out the same way, and any further action dimensions
    (e.g. padding) are ignored.

    Policies normally run through select_action, which handles action chunking and observation history.
    Policies with relative actions (and joint chunk delta actions) instead predict whole chunks, which are converted to
    absolute actions all at once, so that every action of a chunk is relative to the state it was predicted from.
    """

    def __init__(
        self,
        policy,
        preprocessor,
        postprocessor,
        robot: Robot,
        cameras: Dict[str, str],
        optional_cameras: bool = False,
        action_space: str = JOINT_POSITION,
        predict_chunks: bool = False,
        robot_type: str = "",
        invert_gripper: bool = False,
        resize_images: bool = True,
    ):
        """
        Args:
            policy: The lerobot policy (lerobot.policies.pretrained.PreTrainedPolicy).
            preprocessor: The policy's preprocessor pipeline.
            postprocessor: The policy's postprocessor pipeline.
            robot: The robot whose observations and actions to use, e.g. roborollout_vla.robots.droid.ROBOT.
            cameras: Maps observation keys of camera images to the policy's image feature keys.
            optional_cameras: Whether to skip cameras missing from the observation, instead of raising an error.
            action_space: The policy's joint action space: "joint_position", "joint_velocity", or "joint_chunk_delta".
            predict_chunks: Whether to predict whole chunks with predict_action_chunk instead of using select_action.
                Chunks are executed for the policy's n_action_steps. Required for policies with relative actions,
                and for joint_chunk_delta actions.
            robot_type: The robot type passed to the policy, which some policies condition on.
            invert_gripper: Whether the policy's grippers are 1 when open and 0 when closed, the opposite of RoboRollout.
            resize_images: Whether to resize images to the shape of the policy's image features. Otherwise, images
                are passed to the policy at the resolution they're sent at.
        """
        assert action_space in ACTION_SPACES, f"Invalid action space: {action_space}"
        assert predict_chunks or action_space != JOINT_CHUNK_DELTA, "joint_chunk_delta actions require predict_chunks"
        config = policy.config
        if config.env_state_feature is not None:
            raise ValueError("Policies that take an environment state as input are not supported")
        if config.robot_state_feature is not None and config.robot_state_feature.shape[0] < robot.dim:
            raise ValueError(
                f"Expected an observation.state with at least {robot.dim} dims (the {robot.name} robot's joints and "
                f"grippers), got shape {config.robot_state_feature.shape}"
            )
        if config.action_feature is None or config.action_feature.shape[0] < robot.dim:
            raise ValueError(
                f"Expected actions with at least {robot.dim} dims (the {robot.name} robot's joints and grippers), "
                f"got shape {config.action_feature.shape if config.action_feature is not None else None}"
            )
        self._image_shapes = {}
        for feature in cameras.values():
            assert feature in config.image_features, f"Unknown image feature: {feature}"
            self._image_shapes[feature] = tuple(config.image_features[feature].shape[1:])  # (C, H, W) -> (H, W)
        if unused_features := set(config.image_features) - set(cameras.values()):
            logger.warning(f"No cameras are mapped to these image features of the policy: {sorted(unused_features)}")

        self._policy = policy
        self._preprocessor = preprocessor
        self._postprocessor = postprocessor
        self.robot = robot
        self.cameras = cameras
        self.optional_cameras = optional_cameras
        self._warned_missing_cameras = set()
        self.action_space = action_space
        self.predict_chunks = predict_chunks
        self.robot_type = robot_type
        self.invert_gripper = invert_gripper
        self.resize_images = resize_images
        self._device = torch.device(config.device)
        super().__init__(getattr(config, "n_action_steps", None) if predict_chunks else None)

    def reset(self) -> None:
        super().reset()
        self._policy.reset()
        self._preprocessor.reset()
        self._postprocessor.reset()

    def _make_batch(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        """Convert an observation to a batch of size 1, like lerobot's prepare_observation_for_inference."""
        batch = {}
        for obs_key, feature in self.cameras.items():
            if self.optional_cameras and obs_key not in obs:
                if obs_key not in self._warned_missing_cameras:
                    self._warned_missing_cameras.add(obs_key)
                    logger.warning(f"Observation has no camera {obs_key!r}, so skipping image feature {feature}")
                continue
            image = np.asarray(obs[obs_key])
            if self.resize_images:
                image = resize_with_pad(image, *self._image_shapes[feature])
            image = torch.from_numpy(image).to(self._device).to(torch.float32) / 255.0
            batch[feature] = image.permute(2, 0, 1).unsqueeze(0)
        if (state_feature := self._policy.config.robot_state_feature) is not None:
            state = np.zeros(state_feature.shape[0], dtype=np.float32)
            i = 0
            for arm in self.robot.arms:
                state[i : i + arm.n_joints] = obs["qpos"][arm.joints]
                state[i + arm.n_joints] = self._convert_gripper(np.asarray(obs["qpos"][arm.gripper]).item())
                i += arm.n_joints + 1
            batch[STATE_KEY] = torch.from_numpy(state).to(self._device).unsqueeze(0)
        batch["task"] = obs["task"]
        batch["robot_type"] = self.robot_type
        return batch

    def _convert_gripper(self, gripper):
        """Convert gripper positions between RoboRollout's convention and the policy's. This is its own inverse."""
        return 1.0 - gripper if self.invert_gripper else gripper

    def predict_chunk(self, obs: Dict[str, Any]) -> np.ndarray:
        if self._device.type == "cuda" and self._policy.config.use_amp:
            autocast = torch.autocast(device_type=self._device.type)
        else:
            autocast = nullcontext()
        with torch.inference_mode(), autocast:
            batch = self._preprocessor(self._make_batch(obs))
            if self.predict_chunks:
                actions = self._postprocessor(self._policy.predict_action_chunk(batch))  # (1, chunk_len, action_dim)
            else:
                actions = self._postprocessor(self._policy.select_action(batch)).unsqueeze(1)  # (1, 1, action_dim)
        actions = actions[0, :, : self.robot.dim].float().cpu().numpy()
        if self.action_space == JOINT_CHUNK_DELTA:
            i = 0
            for arm in self.robot.arms:
                actions[:, i : i + arm.n_joints] += obs["qpos"][arm.joints]
                i += arm.n_joints + 1
        return actions

    def to_action(self, obs: Dict[str, Any], action: np.ndarray) -> Dict[str, Any]:
        result = {}
        i = 0
        for arm in self.robot.arms:
            joints = action[i : i + arm.n_joints]
            if self.action_space == JOINT_VELOCITY:
                result[arm.joints] = droid.joint_velocity_to_position(obs["qpos"][arm.joints], joints)
            else:
                result[arm.joints] = np.array(joints)
            gripper = action[i + arm.n_joints : i + arm.n_joints + 1]
            result[arm.gripper] = np.clip(self._convert_gripper(gripper), 0.0, 1.0)
            i += arm.n_joints + 1
        return result


def load_policy(path: str, device: Optional[str], n_action_steps: Optional[int]):
    """Load a lerobot policy, its config, and its pre- and postprocessors, like lerobot's rollout does."""
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies import get_policy_class, make_pre_post_processors

    config = PreTrainedConfig.from_pretrained(path)
    config.pretrained_path = path
    if device is not None:
        config.device = device
    if n_action_steps is not None:
        if not hasattr(config, "n_action_steps"):
            raise ValueError(f"Policy type {config.type} doesn't support setting n_action_steps")
        config.n_action_steps = n_action_steps

    policy_cls = get_policy_class(config.type)
    if config.use_peft:
        from peft import PeftConfig, PeftModel

        peft_config = PeftConfig.from_pretrained(path)
        policy = policy_cls.from_pretrained(peft_config.base_model_name_or_path, config=config)
        policy = PeftModel.from_pretrained(policy, path, config=peft_config)
    else:
        policy = policy_cls.from_pretrained(path, config=config)
    # some policies (e.g. MolmoAct2) overwrite n_action_steps with their checkpoint's when they're constructed
    if n_action_steps is not None and policy.config.n_action_steps != n_action_steps:
        logger.info(f"Overriding the policy's n_action_steps of {policy.config.n_action_steps} with {n_action_steps}")
        policy.config.n_action_steps = n_action_steps
    policy = policy.to(config.device)
    policy.eval()

    preprocessor, postprocessor = make_pre_post_processors(
        config,
        pretrained_path=path,
        preprocessor_overrides={"device_processor": {"device": config.device}},
    )
    return policy, config, preprocessor, postprocessor


def main():
    parser = make_parser("Serve a LeRobot policy")
    parser.add_argument("--policy", required=True, help="Local checkpoint directory or HuggingFace repo id")
    parser.add_argument(
        "--robot",
        required=True,
        choices=list(ROBOTS),
        help="Robot to serve the policy to, which sets the observation and action layout (see roborollout_vla.robots)",
    )
    parser.add_argument(
        "--camera",
        action="append",
        default=[],
        metavar="OBS_KEY=FEATURE",
        help="Map a camera in the observation to an image feature of the policy, either its full key "
        "(e.g. observation.images.wrist) or the part after observation.images. (e.g. wrist). Repeat for each camera. "
        "If none are given, each image feature is read from the observation key of the same name, if present.",
    )
    parser.add_argument(
        "--action-space",
        choices=ACTION_SPACES,
        default=JOINT_POSITION,
        help="Action space of the policy's arm joints. joint_position is used by LeRobot's DROID dataset, "
        "and joint_velocity (joint deltas normalized by DROID's 0.2 rad limit) by openpi's DROID policies. "
        "joint_chunk_delta actions are offsets from the joint positions when the chunk was predicted.",
    )
    parser.add_argument(
        "--invert-gripper",
        action="store_true",
        help="Whether the policy's grippers are 1 when open, the opposite of RoboRollout, e.g. for policies trained on "
        "data recorded with i2rt's YAM driver (like MolmoAct2-BimanualYAM).",
    )
    parser.add_argument(
        "--n-action-steps", type=int, help="Actions to execute from each chunk. Defaults to the policy's config."
    )
    parser.add_argument("--device", help="Torch device. Defaults to the policy's config.")
    parser.add_argument("--robot-type", default="", help="Robot type passed to the policy")
    args = parser.parse_args()

    try:
        from lerobot.processor.relative_action_processor import RelativeActionsProcessorStep
    except ImportError as e:
        raise SystemExit("serve-lerobot requires the `lerobot` extra, which requires python 3.12+") from e

    logging.basicConfig(level=logging.INFO, force=True)

    policy, config, preprocessor, postprocessor = load_policy(args.policy, args.device, args.n_action_steps)
    logger.info(f"Loaded {config.type} policy with input features {config.input_features}")

    if args.camera:
        cameras = {}
        for mapping in args.camera:
            obs_key, sep, feature = mapping.partition("=")
            if not sep:
                parser.error(f"--camera must be OBS_KEY=FEATURE, got {mapping!r}")
            cameras[obs_key] = resolve_image_feature(feature, config.image_features)
    else:
        cameras = {feature.removeprefix(IMAGE_PREFIX): feature for feature in config.image_features}

    relative_actions = any(
        isinstance(step, RelativeActionsProcessorStep) and step.enabled for step in preprocessor.steps
    )
    predict_chunks = relative_actions or args.action_space == JOINT_CHUNK_DELTA
    if predict_chunks:
        logger.info("Policy has relative actions, so whole chunks will be predicted and made absolute at once")

    # MolmoAct2's processor resizes images itself, and its image feature shapes are placeholders
    resize_images = config.type != "molmoact2"
    if not resize_images:
        logger.info("Images will be passed to the policy at the resolution they're sent at")

    robot = ROBOTS[args.robot]
    lerobot_policy = LeRobotPolicy(
        policy,
        preprocessor,
        postprocessor,
        robot,
        cameras,
        # without explicit mappings, the observation may have only some of the policy's cameras
        optional_cameras=not args.camera,
        action_space=args.action_space,
        predict_chunks=predict_chunks,
        robot_type=args.robot_type,
        invert_gripper=args.invert_gripper,
        resize_images=resize_images,
    )
    warmup(lerobot_policy, robot.make_dummy_obs(cameras))
    serve(lerobot_policy, args.policy, args, metadata={
        "policy_type": "lerobot",
        "lerobot_policy_type": config.type,
        "cameras": cameras,
        "n_action_steps": getattr(config, "n_action_steps", None),
        "predict_chunks": predict_chunks,
        "resize_images": resize_images,
    })


if __name__ == "__main__":
    main()
