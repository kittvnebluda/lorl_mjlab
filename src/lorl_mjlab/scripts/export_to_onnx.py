"""Export policy to onnx."""

import os
from dataclasses import asdict, dataclass
from pathlib import Path

import tyro
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.rl.runner import MjlabOnPolicyRunner
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

from lorl_mjlab.rl.exporter_utils import export_policy_to_onnx_with_metadata


@dataclass
class Args:
    checkpoint_path: str = "logs/rsl_rl/go1_direction/2026-08-12_23-59-41/model_8999.pt"
    save_path: str = "out/go1_direction/policy.onnx"
    task_id: str = "Lorl-Direction-Rough-Unitree-Go1-Distill"
    obs_group_name: str = "policy"
    device: str = "cpu"


def main():
    args = tyro.cli(Args)
    env_cfg, agent_cfg = load_env_cfg(args.task_id, play=True), load_rl_cfg(args.task_id)
    env = ManagerBasedRlEnv(env_cfg, device=args.device)
    runner_cls = load_runner_cls(args.task_id) or MjlabOnPolicyRunner
    runner = runner_cls(RslRlVecEnvWrapper(env, agent_cfg.clip_actions), asdict(agent_cfg), device=args.device)
    runner.load(args.checkpoint_path, map_location=args.device)

    os.makedirs(Path(args.save_path).parent, exist_ok=True)
    export_policy_to_onnx_with_metadata(
        runner.alg,
        env,
        args.save_path,
        runner.logger,
        upload_model=False,
        obs_group_name=args.obs_group_name,
    )


if __name__ == "__main__":
    main()
