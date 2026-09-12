"""Compare a PyTorch policy rollout against its exported ONNX graph."""

from dataclasses import asdict, dataclass
from typing import Protocol, cast

import numpy as np
import onnxruntime as ort
import torch
import tyro
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.rl.runner import MjlabOnPolicyRunner
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls


@dataclass
class Args:
    checkpoint_path: str = "logs/rsl_rl/go1_direction/2026-08-12_23-59-41/model_8999.pt"
    onnx_path: str = "out/go1_direction/policy.onnx"
    task_id: str = "Lorl-Direction-Rough-Unitree-Go1-Distill"
    steps: int = 32
    seed: int = 0
    tol: float = 1e-5


class OnnxPolicy(Protocol):
    """The ``as_onnx()`` view of a policy, which ``nn.Module`` typing erases."""

    input_names: list[str]
    output_names: list[str]

    def get_dummy_inputs(self) -> tuple[torch.Tensor, ...]: ...

    def __call__(self, *args: torch.Tensor) -> tuple[torch.Tensor | None, ...] | torch.Tensor: ...


def main():
    args = tyro.cli(Args)
    env_cfg, agent_cfg = load_env_cfg(args.task_id, play=True), load_rl_cfg(args.task_id)
    env = ManagerBasedRlEnv(env_cfg, device="cpu")
    runner_cls = load_runner_cls(args.task_id) or MjlabOnPolicyRunner
    runner = runner_cls(RslRlVecEnvWrapper(env, agent_cfg.clip_actions), asdict(agent_cfg), device="cpu")
    runner.load(args.checkpoint_path, map_location="cpu")

    module = cast(OnnxPolicy, runner.alg.get_policy().as_onnx(verbose=False).to("cpu").eval())
    sess = ort.InferenceSession(args.onnx_path, providers=["CPUExecutionProvider"])
    names = module.input_names
    print(f"\n=== {args.task_id} ===\ninputs {names} -> {module.output_names}")

    torch.manual_seed(args.seed)
    dummy = module.get_dummy_inputs()
    state_t = [d.clone() for d in dummy[1:]]  # h_in (+ c_in), zeros at t=0
    state_o = [d.clone().numpy() for d in dummy[1:]]
    obs_dim = dummy[0].shape[-1]

    worst_act, worst_state = 0.0, 0.0
    for t in range(args.steps):
        obs = torch.randn(1, obs_dim)
        with torch.no_grad():
            out = module(obs, *state_t)
        # The GRU head returns a trailing ``None`` where an LSTM would return ``c_out``.
        outs = [o for o in (out if isinstance(out, tuple) else (out,)) if o is not None]
        act_t, state_t = outs[0], outs[1:]

        feeds = {names[0]: obs.numpy(), **{n: s for n, s in zip(names[1:], state_o, strict=True)}}
        res = sess.run(None, feeds)
        act_o, state_o = np.asarray(res[0]), res[1:]

        worst_act = max(worst_act, float(np.abs(act_t.numpy() - act_o).max()))
        if state_t:
            worst_state = max(
                worst_state, max(float(np.abs(a.numpy() - b).max()) for a, b in zip(state_t, state_o, strict=True))
            )
        if t < 3 or t == args.steps - 1:
            print(
                f"  step {t:3d}  |act| max {np.abs(act_o).max():.6f}   act diff {worst_act:.3e}   state diff {worst_state:.3e}"
            )

    print(f"\n  {args.steps} steps: max |action diff| = {worst_act:.3e}, max |state diff| = {worst_state:.3e}")
    print("  PASS" if worst_act < args.tol and worst_state < args.tol else "  FAIL")


if __name__ == "__main__":
    main()
