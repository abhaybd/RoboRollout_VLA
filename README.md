# RoboRollout-VLA

Ready-made policy servers for running off-the-shelf VLAs with [RoboRollout](https://github.com/abhaybd/RoboRollout).
Each server loads a policy, wraps it in RoboRollout's `InferencePolicy`, and serves it with RoboRollout's `WebsocketPolicyServer`. The robot-side rollout script then connects to it as usual.

The following policies are supported, each with its own extra:

| Extra | Policies | Server command |
| --- | --- | --- |
| `openpi` | [openpi](https://github.com/Physical-Intelligence/openpi) DROID policies (`pi0_droid`, `pi0_fast_droid`, `pi05_droid`, ...) | `serve-openpi` |
| `molmobot` | [MolmoBot](https://github.com/allenai/MolmoBot) (e.g. `allenai/MolmoBot-DROID`) | `serve-molmobot` |
| `molmobot-pi0` | [MolmoBot-Pi0](https://github.com/allenai/MolmoBot/tree/main/MolmoBot-Pi0) (e.g. `allenai/MolmoBot-Pi0-DROID`) | `serve-molmobot-pi0` |
| `lerobot` | Any [LeRobot](https://github.com/huggingface/lerobot) policy (ACT, Diffusion, pi0, pi0.5, SmolVLA, GR00T, X-VLA, MolmoAct2, ...) | `serve-lerobot` |

All servers currently follow the observation and action conventions of RoboRollout's DROID robot (see [`droid.py`](src/roborollout_vla/droid.py)).

## Installation

The extras have conflicting dependencies, so install only one per environment.

[uv](https://docs.astral.sh/uv/) is the preferred way to install. From a clone of this repo:

```bash
uv sync --extra openpi  # or molmobot, or molmobot-pi0
uv sync --extra lerobot --python 3.12
```

pip also works (`pip install ".[openpi]"`), but it ignores uv's dependency settings. In particular, openpi depends on packages that only uv knows how to find (`openpi-client`, and a pinned `lerobot`), so pip installs of the `openpi` and `molmobot-pi0` extras need those installed first.

Use Python 3.11 for the MolmoBot and MolmoBot-Pi0 extras, which require it, and Python 3.12+ for the `lerobot` extra, which requires that. The `lerobot` extra installs nothing on Python 3.11.

### PyTorch openpi models

openpi's PyTorch models need patches to `transformers`. They're needed for MolmoBot-Pi0, and for openpi only if you serve a PyTorch checkpoint. After installing, run:

```bash
patch-transformers-for-openpi
```

This modifies the installed `transformers` package, so rerun it if `transformers` is reinstalled.

## Serving a policy

Run the server on the inference machine, then start the rollout on the robot machine with `policy_host`/`policy_port` pointing at it.
Each server loads the policy, runs one warmup inference (which may take a few minutes if the model is compiled), and then serves on port 8000 by default.

```bash
# openpi: pi05_droid from openpi's public checkpoints
serve-openpi --config pi05_droid

# MolmoBot: checkpoints can be a local directory or a HuggingFace repo
serve-molmobot --checkpoint allenai/MolmoBot-DROID

# MolmoBot-Pi0
serve-molmobot-pi0 --checkpoint allenai/MolmoBot-Pi0-DROID

# LeRobot: any policy checkpoint, mapping RoboRollout's cameras to the policy's image features (see below)
serve-lerobot --policy lerobot/MolmoAct2-DROID-LeRobot --camera exo_camera_1=exterior_1_left --camera wrist_camera=wrist_left
```

Run any command with `--help` for all options. Common ones:

- `--host`, `--port`: where to serve the policy.
- `--exo-camera`, `--wrist-camera` (all but LeRobot, see below): observation keys of the exterior and wrist camera images. These must match the keys in the rollout config's `policy_cameras` (`exo_camera_1` and `wrist_camera` by default).
- `--open-loop-horizon` (openpi, MolmoBot-Pi0), `--n-action-steps` (LeRobot): how many actions of each predicted chunk to execute before predicting a new one.

When a client connects, the server sends the model name and metadata, which RoboRollout prints and logs to W&B. The metadata includes the server's command line arguments.

### LeRobot policies

`serve-lerobot` loads any checkpoint saved by LeRobot, along with its pre- and postprocessors, and runs it the way LeRobot's own rollout does. Since LeRobot policies are trained on arbitrary datasets, you need to tell the server how RoboRollout's observations and actions map onto the policy's:

- Each image feature of the policy (`observation.images.<name>`) is read from the observation key `<name>`. Since these names come from the policy's training dataset, set the rollout config's `policy_cameras` to send each camera under the name the policy expects. For example, for `lerobot/MolmoAct2-DROID-LeRobot`:
    ```bash
    python scripts/droid/run_policy.py 'policy_cameras={exterior_1_left: exo_camera_1, wrist_left: wrist_camera}' ...
    ```
    Alternatively, keep the rollout config as is and map cameras on the server with `--camera OBS_KEY=FEATURE`, e.g. `--camera exo_camera_1=exterior_1_left --camera wrist_camera=wrist_left`. Either way, image features the robot doesn't send (or that `--camera` doesn't map) are skipped, which some policies (e.g. pi0/pi0.5) handle by masking the camera out. The server logs the policy's input features on startup.
- `observation.state` is the 7 arm joint positions followed by the gripper position, zero-padded if the policy expects a larger state (e.g. pi0.5's padded 32-dim state).
- The first 8 action dims are the arm and gripper, and the rest (e.g. padding) are ignored. `--action-space` sets what the arm actions are:
    - `joint_position` (default): absolute joint positions, as in LeRobot's port of the DROID dataset.
    - `joint_velocity`: DROID's joint velocities, i.e. joint deltas divided by DROID's 0.2 rad limit, as used by openpi's DROID policies (and their LeRobot ports).
    - `joint_chunk_delta`: offsets from the joint positions when the chunk was predicted.

Policies whose processors convert relative actions back to absolute ones, and `joint_chunk_delta` policies, predict whole chunks at once, so that every action of a chunk is relative to the state it was predicted from. Policies that take an environment state as input aren't supported.

## Adding a policy

1. Add an extra with the policy's package to [`pyproject.toml`](pyproject.toml), and add it to the `conflicts` list in `[tool.uv]` if it can't be installed alongside the others.
2. Add a module under [`src/roborollout_vla`](src/roborollout_vla) with:
    - An `InferencePolicy` subclass that converts RoboRollout observations to the policy's inputs, and the policy's outputs to RoboRollout actions. Policies that predict action chunks can subclass [`ChunkedPolicy`](src/roborollout_vla/chunking.py).
    - A `main()` that parses arguments with `make_parser()`, loads the policy, and calls `warmup()` and `serve()` from [`server.py`](src/roborollout_vla/server.py). Import the policy's package inside `main()`, so the wrapper can be tested without it.
3. Register `main()` as a script in `[project.scripts]`.

## Development

```bash
uv run pytest
```

The tests use fake policies, so they don't need any extras, though the LeRobot tests are skipped unless PyTorch is installed.
