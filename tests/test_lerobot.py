"""
Tests of the lerobot policy wrapper, using a fake policy and processors so that lerobot isn't needed.
"""

from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from roborollout_vla import droid  # noqa: E402
from roborollout_vla.lerobot import (  # noqa: E402
    JOINT_CHUNK_DELTA,
    JOINT_POSITION,
    JOINT_VELOCITY,
    LeRobotPolicy,
    resolve_image_feature,
)

EXO_FEATURE = "observation.images.exterior"
WRIST_FEATURE = "observation.images.wrist"


def feature(*shape):
    return SimpleNamespace(shape=shape)


class FakeProcessor:
    """Records its inputs, and applies fn to them."""

    def __init__(self, fn=lambda x: x):
        self.fn = fn
        self.inputs = []
        self.resets = 0

    def __call__(self, x):
        self.inputs.append(x)
        return self.fn(x)

    def reset(self):
        self.resets += 1


class FakeLeRobotPolicy:
    """Predicts chunks whose actions are all chunk_idx + step / 10, with `action_dim` dims."""

    def __init__(self, state_dim=8, action_dim=8, chunk_size=4, n_action_steps=3, image_shape=(3, 180, 320)):
        self.config = SimpleNamespace(
            image_features={EXO_FEATURE: feature(*image_shape), WRIST_FEATURE: feature(*image_shape)},
            robot_state_feature=feature(state_dim),
            env_state_feature=None,
            action_feature=feature(action_dim),
            n_action_steps=n_action_steps,
            device="cpu",
            use_amp=False,
        )
        self.chunk_size = chunk_size
        self.chunks_predicted = 0
        self.resets = 0
        self._queue = []

    def reset(self):
        self.resets += 1
        self._queue = []

    def predict_action_chunk(self, batch):
        steps = torch.arange(self.chunk_size, dtype=torch.float32) / 10 + self.chunks_predicted
        self.chunks_predicted += 1
        return steps[None, :, None].expand(1, self.chunk_size, self.config.action_feature.shape[0]).clone()

    def select_action(self, batch):
        if not self._queue:
            self._queue = list(self.predict_action_chunk(batch)[0, : self.config.n_action_steps])
        return self._queue.pop(0)[None]


def make_policy(fake=None, action_space=JOINT_POSITION, predict_chunks=False, cameras=None):
    fake = fake or FakeLeRobotPolicy()
    cameras = cameras or {droid.EXO_CAMERA: EXO_FEATURE, droid.WRIST_CAMERA: WRIST_FEATURE}
    pre, post = FakeProcessor(), FakeProcessor()
    policy = LeRobotPolicy(fake, pre, post, cameras, action_space=action_space, predict_chunks=predict_chunks)
    return policy, fake, pre, post


def make_obs(arm=None):
    obs = droid.make_dummy_obs([droid.EXO_CAMERA, droid.WRIST_CAMERA], task="put the mug in the bowl")
    obs[droid.EXO_CAMERA][:] = 255
    obs["qpos"]["arm"] = np.zeros(7) if arm is None else np.asarray(arm, dtype=float)
    obs["qpos"]["gripper"] = np.array([0.5])
    return obs


def test_resolve_image_feature():
    features = {EXO_FEATURE: feature(3, 1, 1)}
    assert resolve_image_feature(EXO_FEATURE, features) == EXO_FEATURE
    assert resolve_image_feature("exterior", features) == EXO_FEATURE
    with pytest.raises(ValueError, match="exterior"):
        resolve_image_feature("wrist", features)


def test_batch():
    policy, _, pre, _ = make_policy(FakeLeRobotPolicy(state_dim=32))
    policy.get_action(make_obs(arm=np.arange(7)))

    (batch,) = pre.inputs
    assert batch["task"] == "put the mug in the bowl"
    # images are resized to the feature shape, converted to CHW in [0, 1], and batched
    assert batch[EXO_FEATURE].shape == (1, 3, 180, 320)
    assert batch[EXO_FEATURE].dtype == torch.float32
    assert batch[EXO_FEATURE].max() == 1.0
    assert batch[WRIST_FEATURE].max() == 0.0
    # the state is zero-padded to the feature shape
    expected_state = np.zeros(32, dtype=np.float32)
    expected_state[:8] = [*range(7), 0.5]
    np.testing.assert_array_equal(batch["observation.state"][0].numpy(), expected_state)


def test_select_action_is_called_every_step():
    policy, fake, pre, post = make_policy()
    arms = [policy.get_action(make_obs())["arm"][0] for _ in range(4)]
    np.testing.assert_allclose(arms, [0, 0.1, 0.2, 1])  # the fake policy executes n_action_steps=3 per chunk
    assert len(pre.inputs) == len(post.inputs) == 4

    policy.reset()
    assert fake.resets == pre.resets == post.resets == 2  # also reset on construction


def test_predict_chunks_postprocesses_whole_chunks():
    policy, _, pre, post = make_policy(predict_chunks=True)
    arms = [policy.get_action(make_obs())["arm"][0] for _ in range(4)]
    # chunks are executed for n_action_steps, not the whole chunk_size
    np.testing.assert_allclose(arms, [0, 0.1, 0.2, 1])
    assert len(pre.inputs) == len(post.inputs) == 2
    assert post.inputs[0].shape == (1, 4, 8)


def test_joint_chunk_delta_actions_are_relative_to_chunk_start():
    policy, _, _, _ = make_policy(action_space=JOINT_CHUNK_DELTA, predict_chunks=True)
    first = policy.get_action(make_obs(arm=np.ones(7)))
    # the arm moves, but the rest of the chunk stays relative to the state it was predicted from
    second = policy.get_action(make_obs(arm=np.full(7, 5.0)))
    np.testing.assert_allclose(first["arm"], np.ones(7))
    np.testing.assert_allclose(second["arm"], np.full(7, 1.1))

    with pytest.raises(AssertionError):
        make_policy(action_space=JOINT_CHUNK_DELTA, predict_chunks=False)


def test_extra_action_dims_are_ignored():
    policy, _, _, _ = make_policy(FakeLeRobotPolicy(action_dim=32), action_space=JOINT_VELOCITY)
    action = policy.get_action(make_obs())
    np.testing.assert_allclose(action["arm"], np.zeros(7))
    np.testing.assert_allclose(action["gripper"], [0.0])


def test_incompatible_features_are_rejected():
    with pytest.raises(ValueError, match="observation.state"):
        make_policy(FakeLeRobotPolicy(state_dim=6))
    with pytest.raises(ValueError, match="actions"):
        make_policy(FakeLeRobotPolicy(action_dim=7))


def test_missing_cameras_are_skipped_only_if_optional():
    obs = make_obs()
    del obs[droid.WRIST_CAMERA]

    policy, _, pre, _ = make_policy()
    with pytest.raises(KeyError):
        policy.get_action(obs)

    fake = FakeLeRobotPolicy()
    pre = FakeProcessor()
    cameras = {droid.EXO_CAMERA: EXO_FEATURE, droid.WRIST_CAMERA: WRIST_FEATURE}
    policy = LeRobotPolicy(fake, pre, FakeProcessor(), cameras, optional_cameras=True)
    policy.get_action(obs)
    assert EXO_FEATURE in pre.inputs[0]
    assert WRIST_FEATURE not in pre.inputs[0]
