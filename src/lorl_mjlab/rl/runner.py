"""RSL-RL runner extensions.

``mjlab.rl.exporter_utils.get_base_metadata`` hardcodes the actor-facing
observation group name as ``"actor"``. This task's asymmetric groups are named
``"policy"``/``"privileged"`` (matching the source task's naming), so this
module provides a local, parametrized copy plus a distillation runner with
the same checkpoint/env-state persistence and ONNX export conveniences as
``mjlab.rl.runner.MjlabOnPolicyRunner``.
"""

import os
from pathlib import Path

import onnx
import torch
import wandb
from mjlab.entity import Entity
from mjlab.envs import ManagerBasedRlEnv
from mjlab.envs.mdp.actions import JointPositionAction
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.rl.exporter_utils import list_to_csv_str
from mjlab.rl.runner import MjlabOnPolicyRunner
from rsl_rl.runners import DistillationRunner


def get_direction_metadata(
    env: ManagerBasedRlEnv, run_path: str, obs_group_name: str = "policy"
) -> dict[str, list | str | float]:
    """Same as ``mjlab.rl.exporter_utils.get_base_metadata``, but reads the
    deployed observation group by name instead of assuming it's ``"actor"``."""
    robot: Entity = env.scene["robot"]
    joint_action = env.action_manager.get_term("joint_pos")
    assert isinstance(joint_action, JointPositionAction)
    joint_name_to_ctrl_id = {}
    for actuator in robot.spec.actuators:
        joint_name = actuator.target.split("/")[-1]
        joint_name_to_ctrl_id[joint_name] = actuator.id
    ctrl_ids_natural = [joint_name_to_ctrl_id[jname] for jname in robot.joint_names if jname in joint_name_to_ctrl_id]
    joint_stiffness = env.sim.mj_model.actuator_gainprm[ctrl_ids_natural, 0]
    joint_damping = -env.sim.mj_model.actuator_biasprm[ctrl_ids_natural, 2]
    return {
        "run_path": run_path,
        "joint_names": list(robot.joint_names),
        "joint_stiffness": joint_stiffness.tolist(),
        "joint_damping": joint_damping.tolist(),
        "default_joint_pos": robot.data.default_joint_pos[0].cpu().tolist(),
        "command_names": list(env.command_manager.active_terms),
        "observation_names": env.observation_manager.active_terms[obs_group_name],
        "action_scale": joint_action._scale[0].cpu().tolist()
        if isinstance(joint_action._scale, torch.Tensor)
        else joint_action._scale,
    }


def _attach_metadata_to_onnx(onnx_path: str, metadata: dict[str, list | str | float]) -> None:
    model = onnx.load(onnx_path)
    for k, v in metadata.items():
        entry = onnx.StringStringEntryProto()
        entry.key = k
        entry.value = list_to_csv_str(v) if isinstance(v, list) else str(v)
        model.metadata_props.append(entry)
    onnx.save(model, onnx_path)


def _get_export_paths(checkpoint_path: str) -> tuple[Path, str, Path]:
    export_dir = Path(checkpoint_path).parent
    filename = f"{export_dir.name}.onnx"
    return export_dir, filename, export_dir / filename


def _export_policy_and_attach_metadata(
    alg, env: ManagerBasedRlEnv, path: str, logger, upload_model: bool, obs_group_name: str
) -> None:
    export_dir, filename, onnx_path = _get_export_paths(path)
    onnx_model = alg.get_policy().as_onnx(verbose=False)
    onnx_model.to("cpu")
    onnx_model.eval()
    os.makedirs(str(export_dir), exist_ok=True)
    torch.onnx.export(
        onnx_model,
        onnx_model.get_dummy_inputs(),
        str(onnx_path),
        export_params=True,
        opset_version=18,
        input_names=onnx_model.input_names,
        output_names=onnx_model.output_names,
        dynamic_axes={},
        dynamo=False,
    )
    run_name = (wandb.run.name if logger.logger_type == "wandb" and wandb.run else "local") or "local"
    metadata = get_direction_metadata(env, run_name, obs_group_name)
    _attach_metadata_to_onnx(str(onnx_path), metadata)
    if logger.logger_type == "wandb" and upload_model:
        wandb.save(str(onnx_path), base_path=str(export_dir))


class LorlOnPolicyRunner(MjlabOnPolicyRunner):
    """PPO runner exporting ONNX metadata from the "policy" obs group."""

    env: RslRlVecEnvWrapper

    def save(self, path: str, infos=None) -> None:
        super().save(path, infos)
        try:
            _export_policy_and_attach_metadata(
                self.alg,
                self.env.unwrapped,
                path,
                self.logger,
                self.cfg["upload_model"],
                obs_group_name="policy",
            )
        except Exception as e:
            print(f"[WARN] ONNX export failed (training continues): {e}")


class LorlDistillationRunner(DistillationRunner):
    """Distillation runner with mjlab's checkpoint/env-state persistence and
    ONNX export, exporting the deployable student policy from the "policy" group."""

    env: RslRlVecEnvWrapper

    def __init__(self, env, train_cfg: dict, log_dir: str | None = None, device: str = "cpu") -> None:
        for key in ("student", "teacher"):
            if key in train_cfg:
                for opt in ("cnn_cfg", "distribution_cfg"):
                    if train_cfg[key].get(opt) is None:
                        train_cfg[key].pop(opt, None)
                if train_cfg[key].get("rnn_type") is None:
                    for opt in ("rnn_type", "rnn_hidden_dim", "rnn_num_layers"):
                        train_cfg[key].pop(opt, None)
        super().__init__(env, train_cfg, log_dir, device)

    def save(self, path: str, infos=None) -> None:
        env_state = {"common_step_counter": self.env.unwrapped.common_step_counter}
        infos = {**(infos or {}), "env_state": env_state}
        saved_dict = self.alg.save()
        saved_dict["iter"] = self.current_learning_iteration
        saved_dict["infos"] = infos
        torch.save(saved_dict, path)
        if self.cfg["upload_model"]:
            self.logger.save_model(path, self.current_learning_iteration)

        try:
            _export_policy_and_attach_metadata(
                self.alg,
                self.env.unwrapped,
                path,
                self.logger,
                self.cfg["upload_model"],
                obs_group_name="policy",
            )
        except Exception as e:
            print(f"[WARN] ONNX export failed (training continues): {e}")

    def load(
        self,
        path: str,
        load_cfg: dict | None = None,
        strict: bool = True,
        map_location: str | None = None,
    ) -> dict:
        loaded_dict = torch.load(path, map_location=map_location, weights_only=False)
        load_iteration = self.alg.load(loaded_dict, load_cfg, strict)
        if load_iteration:
            self.current_learning_iteration = loaded_dict["iter"]
        infos = loaded_dict["infos"]
        if infos and "env_state" in infos:
            self.env.unwrapped.common_step_counter = infos["env_state"]["common_step_counter"]
        return infos
