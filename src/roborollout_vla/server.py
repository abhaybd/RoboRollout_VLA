"""
Shared entry point logic for serving a policy with RoboRollout's WebsocketPolicyServer.
"""

import argparse
import logging
from typing import Any, Dict, Optional

from robo_rollout.policy.policy_server import InferencePolicy, WebsocketPolicyServer

logger = logging.getLogger(__name__)


def make_parser(description: str) -> argparse.ArgumentParser:
    """Create an argument parser with the options common to all policy servers."""
    parser = argparse.ArgumentParser(description=description, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--host", default="0.0.0.0", help="Host to serve the policy on")
    parser.add_argument("--port", type=int, default=8000, help="Port to serve the policy on")
    return parser


def warmup(policy: InferencePolicy, obs: Dict[str, Any]) -> None:
    """
    Run inference once and reset the policy, so that slow first-call work (e.g. compilation) doesn't
    happen while a robot is waiting on the policy.
    """
    logger.info("Warming up policy...")
    policy.get_action(obs)
    policy.reset()
    logger.info("Warmup complete")


def serve(
    policy: InferencePolicy,
    model_name: str,
    args: argparse.Namespace,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Serve a policy until interrupted.

    Args:
        policy: The policy to serve.
        model_name: The name of the model, sent to clients in the metadata.
        args: The parsed command line arguments, which are included in the metadata (except host and port).
        metadata: Additional metadata to send to clients. Must be msgpack-serializable.
    """
    server_metadata = {k: v for k, v in vars(args).items() if k not in ("host", "port")}
    server_metadata.update(metadata or {})
    server = WebsocketPolicyServer(policy, model_name, host=args.host, port=args.port, metadata=server_metadata)
    logger.info(f"Serving {model_name} on {args.host}:{args.port}")
    server.serve_forever()
