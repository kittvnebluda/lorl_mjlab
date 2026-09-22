import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
import wandb
from mjlab.actuator import IdealPdActuator
from mjlab.entity import Entity
from mjlab.envs import ManagerBasedRlEnv
from mjlab.envs.mdp.actions import JointPositionAction
from mjlab.rl.exporter_utils import list_to_csv_str
from rsl_rl.algorithms import PPO, Distillation
from rsl_rl.utils.logger import Logger


def _deploy_gains(
    env: ManagerBasedRlEnv, robot: Entity, ctrl_ids_natural: list[int]
) -> tuple[list[float], list[float]]:
    """PD gains a deployment controller should run, per joint in natural order.

    Builtin position actuators keep their gains in the compiled model, so read them from
    there. Explicit-torque actuators (``IdealPdActuator`` and its subclasses, including the
    learned MLP model) evaluate their control law in Python and compile down to ``<motor>``
    elements whose gainprm/biasprm carry no gains at all, so take those from the actuator
    objects instead -- for the learned model those are the gains its network behaves like.
    """
    kp = env.sim.mj_model.actuator_gainprm[:, 0].copy()
    kd = -env.sim.mj_model.actuator_biasprm[:, 2].copy()
    for actuator in robot.actuators:
        if not isinstance(actuator, IdealPdActuator):
            continue
        assert actuator.stiffness is not None and actuator.damping is not None
        ctrl_ids = actuator.global_ctrl_ids.cpu().numpy()
        kp[ctrl_ids] = actuator.stiffness[0].cpu().numpy()
        kd[ctrl_ids] = actuator.damping[0].cpu().numpy()
    return kp[ctrl_ids_natural].tolist(), kd[ctrl_ids_natural].tolist()


def get_common_metadata(
    env: ManagerBasedRlEnv, run_path: str, obs_group_name: str = "policy"
) -> dict[str, list | str | float]:
    """Metadata every deployment needs, whatever the action space.

    Joint names, deployment PD gains, the default pose and the observation layout: everything a
    controller must know to build the policy's input vector and drive the joints. What it does
    *not* include is how the policy's output becomes joint targets -- that is action-space
    specific, and each exporter adds it on top.
    """
    robot: Entity = env.scene["robot"]
    joint_name_to_ctrl_id = {}
    for actuator in robot.spec.actuators:
        joint_name = actuator.target.split("/")[-1]
        joint_name_to_ctrl_id[joint_name] = actuator.id
    ctrl_ids_natural = [joint_name_to_ctrl_id[jname] for jname in robot.joint_names if jname in joint_name_to_ctrl_id]
    joint_stiffness, joint_damping = _deploy_gains(env, robot, ctrl_ids_natural)

    observation_term_scale: list = []
    observation_term_flatten_history_dim: list = []
    observation_term_history_length: list = []
    observation_term_clip: list = []
    observation_names = env.observation_manager.active_terms[obs_group_name]

    for active_term in observation_names:
        cfg = env.observation_manager.get_term_cfg(obs_group_name, active_term)

        if cfg.scale is None:
            observation_term_scale.append(1.0)
        else:
            raw_scale = cfg.scale
            scale = raw_scale.cpu().tolist() if isinstance(raw_scale, torch.Tensor) else raw_scale
            observation_term_scale.append(scale)

        raw_clip = cfg.clip
        if raw_clip is None:
            observation_term_clip.append([float("-inf"), float("inf")])
        else:
            observation_term_clip.append(list(raw_clip))

        observation_term_flatten_history_dim.append(cfg.flatten_history_dim)
        observation_term_history_length.append(cfg.history_length)

    return {
        "run_path": run_path,
        "joint_names": list(robot.joint_names),
        "joint_stiffness": joint_stiffness,
        "joint_damping": joint_damping,
        "default_joint_pos": robot.data.default_joint_pos[0].cpu().tolist(),
        "command_names": list(env.command_manager.active_terms),
        "observation_names": observation_names,
        "observation_terms_scale": observation_term_scale,
        "observation_terms_flatten_history_dim": observation_term_flatten_history_dim,
        "observation_terms_history_length": observation_term_history_length,
        "observation_terms_clip": observation_term_clip,
    }


def get_base_metadata(
    env: ManagerBasedRlEnv, run_path: str, obs_group_name: str = "policy"
) -> dict[str, list | str | float]:
    """Metadata for a joint-position policy: the common fields plus the action scale."""
    joint_action = env.action_manager.get_term("joint_pos")
    assert isinstance(joint_action, JointPositionAction)
    metadata = get_common_metadata(env, run_path, obs_group_name)
    metadata["action_scale"] = (
        joint_action._scale[0].cpu().tolist() if isinstance(joint_action._scale, torch.Tensor) else joint_action._scale
    )
    return metadata


def _attach_metadata_to_onnx(program: torch.onnx.ONNXProgram, metadata: dict[str, list | str | float]) -> None:
    """Stamp metadata onto the in-memory IR model, which is what ``ONNXProgram.save`` serializes.

    ``program.model_proto`` builds a throwaway proto on each access, so mutating it is a no-op.
    """
    for k, v in metadata.items():
        program.model.metadata_props[k] = list_to_csv_str(v) if isinstance(v, list) else str(v)


def export_policy_to_onnx_with_metadata(
    alg: PPO | Distillation,
    env: ManagerBasedRlEnv,
    path: str,
    logger: Logger,
    upload_model: bool,
    obs_group_name: str,
    metadata_fn: Callable[[ManagerBasedRlEnv, str, str], dict[str, list | str | float]] = get_base_metadata,
) -> None:
    onnx_path = Path(path).with_suffix(".onnx")
    export_dir = onnx_path.parent
    onnx_model: Any = alg.get_policy().as_onnx(verbose=False).to("cpu").eval()
    os.makedirs(str(export_dir), exist_ok=True)

    onnx_program = torch.onnx.export(
        onnx_model,
        onnx_model.get_dummy_inputs(),
        opset_version=18,
        input_names=onnx_model.input_names,
        output_names=onnx_model.output_names,
        optimize=True,
        verbose=False,
    )
    assert onnx_program is not None, "Something went wrong during model export"

    run_name = (wandb.run.name if logger.logger_type == "wandb" and wandb.run else "local") or "local"
    metadata = metadata_fn(env, run_name, obs_group_name)
    _attach_metadata_to_onnx(onnx_program, metadata)
    onnx_program.save(str(onnx_path), external_data=False)

    if logger.logger_type == "wandb" and upload_model:
        wandb.save(str(onnx_path), base_path=str(export_dir))
