"""
Tests of the policy wrappers, using fake underlying policies so that no policy extras are needed.
"""

import argparse
import socket
import threading
import time

import numpy as np
import pytest

from robo_rollout.policy.policy_client import PolicyClient

from roborollout_vla import droid
from roborollout_vla.molmobot import MolmoBotPolicy
from roborollout_vla.molmobot_pi0 import MolmoBotPi0Policy
from roborollout_vla.openpi import JOINT_POSITION, JOINT_VELOCITY, OpenPIPolicy
from roborollout_vla.server import serve


class FakeOpenPIPolicy:
    """Records inputs and returns the next chunk from a list."""

    def __init__(self, chunks):
        self.chunks = list(chunks)
        self.inputs = []

    def infer(self, obs):
        self.inputs.append(obs)
        return {"actions": self.chunks[len(self.inputs) - 1]}


class FakeMolmoBotPolicy:
    def __init__(self, action):
        self.action = action
        self.inputs = []
        self.resets = 0

    def reset(self):
        self.resets += 1

    def get_action(self, obs):
        self.inputs.append(obs)
        return self.action


def make_obs(arm=None, gripper=0.5, exo_camera=droid.EXO_CAMERA, wrist_camera=droid.WRIST_CAMERA):
    obs = droid.make_dummy_obs([exo_camera, wrist_camera], task="put the mug in the bowl")
    obs["qpos"]["arm"] = np.zeros(7) if arm is None else np.asarray(arm, dtype=float)
    obs["qpos"]["gripper"] = np.array([gripper])
    return obs


def test_joint_velocity_to_position_rescales_large_velocities():
    qpos = np.ones(7)
    np.testing.assert_allclose(droid.joint_velocity_to_position(qpos, np.full(7, 0.5)), np.full(7, 1.1))
    # the largest velocity is scaled to 1, preserving the direction
    velocity = np.array([2.0, -1.0, 0, 0, 0, 0, 0])
    expected = qpos + np.array([1.0, -0.5, 0, 0, 0, 0, 0]) * droid.MAX_JOINT_DELTA
    np.testing.assert_allclose(droid.joint_velocity_to_position(qpos, velocity), expected)


def test_to_molmospaces_gripper():
    np.testing.assert_allclose(droid.to_molmospaces_gripper(np.array([1.0])), np.full(2, droid.MOLMOSPACES_GRIPPER_CLOSED))
    np.testing.assert_allclose(droid.to_molmospaces_gripper(np.array([0.0])), np.zeros(2))


def test_chunks_are_executed_for_open_loop_horizon():
    chunks = [np.full((10, 8), i, dtype=float) for i in range(3)]
    fake = FakeOpenPIPolicy(chunks)
    policy = OpenPIPolicy(fake, action_space=JOINT_POSITION, open_loop_horizon=4)
    arms = [policy.get_action(make_obs())["arm"][0] for _ in range(6)]
    assert arms == [0, 0, 0, 0, 1, 1]
    assert len(fake.inputs) == 2

    # reset discards the current chunk
    policy.reset()
    assert policy.get_action(make_obs())["arm"][0] == 2


def test_whole_chunk_is_executed_without_open_loop_horizon():
    chunks = [np.full((3, 8), i, dtype=float) for i in range(2)]
    policy = MolmoBotPi0Policy(FakeOpenPIPolicy(chunks), joint_velocity=False)
    arms = [policy.get_action(make_obs())["arm"][0] for _ in range(4)]
    assert arms == [0, 0, 0, 1]


def test_openpi_inputs():
    fake = FakeOpenPIPolicy([np.zeros((10, 8))])
    policy = OpenPIPolicy(fake, exo_camera="exo", wrist_camera="wrist")
    obs = make_obs(arm=np.arange(7), gripper=0.25, exo_camera="exo", wrist_camera="wrist")
    obs["wrist"] = np.ones((360, 640, 3), dtype=np.uint8)
    policy.get_action(obs)

    (inputs,) = fake.inputs
    assert inputs["prompt"] == "put the mug in the bowl"
    np.testing.assert_array_equal(inputs["observation/joint_position"], np.arange(7))
    np.testing.assert_array_equal(inputs["observation/gripper_position"], [0.25])
    assert inputs["observation/exterior_image_1_left"] is obs["exo"]
    assert inputs["observation/wrist_image_left"] is obs["wrist"]


def test_openpi_joint_velocity_actions():
    chunk = np.zeros((10, 8))
    chunk[0, :7] = [0.5, 2.0, -3.0, 0, 0, 0, 0]  # clipped to [-1, 1], like openpi's DROID client
    chunk[0, 7] = 0.6
    chunk[1, 7] = 0.4
    policy = OpenPIPolicy(FakeOpenPIPolicy([chunk]), action_space=JOINT_VELOCITY)
    qpos = np.arange(7, dtype=float)

    action = policy.get_action(make_obs(arm=qpos))
    np.testing.assert_allclose(action["arm"], qpos + np.array([0.5, 1, -1, 0, 0, 0, 0]) * droid.MAX_JOINT_DELTA)
    np.testing.assert_array_equal(action["gripper"], [1.0])  # binarized
    np.testing.assert_array_equal(policy.get_action(make_obs(arm=qpos))["gripper"], [0.0])


def test_molmobot_pi0_inputs_and_actions():
    chunk = np.zeros((10, 8))
    chunk[0, :7] = [2.0, -1.0, 0, 0, 0, 0, 0]
    chunk[0, 7] = 1.3
    fake = FakeOpenPIPolicy([chunk])
    policy = MolmoBotPi0Policy(fake, joint_velocity=True)
    obs = make_obs(gripper=1.0)
    obs[droid.EXO_CAMERA] = np.zeros((720, 1280, 3), dtype=np.uint8)
    action = policy.get_action(obs)

    (inputs,) = fake.inputs
    assert inputs["observation/exterior_image_1_left"].shape == (224, 224, 3)
    assert inputs["observation/wrist_image_left"].shape == (224, 224, 3)
    np.testing.assert_allclose(inputs["observation/gripper_position"], np.full(2, droid.MOLMOSPACES_GRIPPER_CLOSED))
    # velocities are rescaled rather than clipped, and the gripper is clipped but not binarized
    np.testing.assert_allclose(action["arm"], np.array([1.0, -0.5, 0, 0, 0, 0, 0]) * droid.MAX_JOINT_DELTA)
    np.testing.assert_array_equal(action["gripper"], [1.0])


def test_molmobot_inputs_and_actions():
    fake = FakeMolmoBotPolicy({"arm": np.arange(7, dtype=float), "gripper": np.array([255.0])})
    policy = MolmoBotPolicy(fake, exo_camera="exo", wrist_camera="wrist")
    obs = make_obs(gripper=0.5, exo_camera="exo", wrist_camera="wrist")
    action = policy.get_action(obs)

    (inputs,) = fake.inputs
    assert inputs["exo_camera_1"] is obs["exo"]
    assert inputs["wrist_camera"] is obs["wrist"]
    np.testing.assert_allclose(inputs["qpos"]["gripper"], np.full(2, 0.5 * droid.MOLMOSPACES_GRIPPER_CLOSED))
    np.testing.assert_array_equal(action["arm"], np.arange(7))
    np.testing.assert_array_equal(action["gripper"], [1.0])

    policy.reset()
    assert fake.resets == 1


def test_serve_end_to_end():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    policy = OpenPIPolicy(FakeOpenPIPolicy([np.ones((10, 8))]), action_space=JOINT_POSITION)
    args = argparse.Namespace(host="127.0.0.1", port=port, config="fake_config")
    threading.Thread(
        target=serve, args=(policy, "fake_model", args), kwargs={"metadata": {"extra": 1}}, daemon=True
    ).start()
    # wait for the server to come up, since PolicyClient waits 5s between connection attempts
    deadline = time.monotonic() + 5
    while True:
        try:
            socket.create_connection(("127.0.0.1", port)).close()
            break
        except ConnectionRefusedError:
            assert time.monotonic() < deadline, "server did not start"
            time.sleep(0.01)

    client = PolicyClient(host="127.0.0.1", port=port)
    assert client.get_server_metadata() == {"config": "fake_config", "extra": 1, "model_name": "fake_model"}
    action = client.infer(make_obs())
    client.close()
    np.testing.assert_array_equal(action["arm"], np.ones(7))
    np.testing.assert_array_equal(action["gripper"], [1.0])
