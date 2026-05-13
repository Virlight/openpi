"""Serve a text-CFG policy without modifying the default server script."""

import dataclasses
import logging
import socket

import tyro

from openpi.policies import policy as _policy
from openpi.policies import policy_config as _policy_config
from openpi.serving import websocket_policy_server
from openpi.training import config as _base_config
import openpi.training_stage2.config as _stage2_config


@dataclasses.dataclass
class Args:
    config: str = "pi05_maniparena_ee_stage2_text_cfg"
    checkpoint_dir: str = "checkpoints/pi05_maniparena_ee_stage2_text_cfg"
    default_prompt: str | None = None
    text_cfg_scale: float = 1.5
    port: int = 8000
    record: bool = False


def get_config(config_name: str):
    for config_module in (_stage2_config, _base_config):
        try:
            return config_module.get_config(config_name)
        except ValueError:
            pass
    raise ValueError(f"Text-CFG config '{config_name}' not found.")


def create_policy(args: Args) -> _policy.Policy:
    return _policy_config.create_trained_policy(
        get_config(args.config),
        args.checkpoint_dir,
        default_prompt=args.default_prompt,
        sample_kwargs={"text_cfg_scale": args.text_cfg_scale},
    )


def main(args: Args) -> None:
    policy = create_policy(args)
    policy_metadata = policy.metadata
    if args.record:
        policy = _policy.PolicyRecorder(policy, "policy_records")

    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logging.info("Creating server (host: %s, ip: %s)", hostname, local_ip)

    server = websocket_policy_server.WebsocketPolicyServer(
        policy=policy,
        host="0.0.0.0",
        port=args.port,
        metadata=policy_metadata,
    )
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
