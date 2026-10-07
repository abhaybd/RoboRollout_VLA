from abc import abstractmethod
from typing import Any, Dict, Optional

import numpy as np

from robo_rollout.policy.policy_server import InferencePolicy


class ChunkedPolicy(InferencePolicy):
    """
    Adapts a policy that predicts chunks of actions to RoboRollout, which requests one action per step.
    Executes up to open_loop_horizon actions of each chunk before predicting a new one.
    """

    def __init__(self, open_loop_horizon: Optional[int] = None):
        """
        Args:
            open_loop_horizon: How many actions of each chunk to execute. If None, executes the whole chunk.
        """
        assert open_loop_horizon is None or open_loop_horizon > 0, \
            f"open_loop_horizon must be null or positive, got {open_loop_horizon}"
        self.open_loop_horizon = open_loop_horizon
        self.reset()

    def reset(self) -> None:
        self._chunk: Optional[np.ndarray] = None
        self._chunk_idx = 0

    @abstractmethod
    def predict_chunk(self, obs: Dict[str, Any]) -> np.ndarray:
        """Predict a chunk of actions, with shape (chunk_len, action_dim)."""
        pass

    @abstractmethod
    def to_action(self, obs: Dict[str, Any], action: np.ndarray) -> Dict[str, Any]:
        """Convert one action of a chunk to an action for the client. obs is the current observation."""
        pass

    def _chunk_exhausted(self) -> bool:
        if self._chunk is None or self._chunk_idx >= len(self._chunk):
            return True
        return self.open_loop_horizon is not None and self._chunk_idx >= self.open_loop_horizon

    def get_action(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        if self._chunk_exhausted():
            self._chunk = np.asarray(self.predict_chunk(obs))
            self._chunk_idx = 0
        action = self._chunk[self._chunk_idx]
        self._chunk_idx += 1
        return self.to_action(obs, action)
