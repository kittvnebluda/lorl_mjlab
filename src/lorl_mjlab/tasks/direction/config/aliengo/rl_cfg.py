"""RL configuration for Unitree AlienGo direction task."""

from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg

from lorl_mjlab.rl.config import (
    RslRlDistillationAlgorithmCfg,
    RslRlDistillationRunnerCfg,
    RslRlSymmetricPpoAlgorithmCfg,
    RslRlSymmetryCfg,
)
from lorl_mjlab.tasks.direction.mdp.symmetry import compute_symmetric_states

# The PPO actor is a privileged "teacher": it reads both the policy (proprio)
# and privileged (teacher-only) observation groups, matching the source
# IsaacLab training setup. The Distill variant later trains a proprio-only
# "student" to imitate this teacher.
_TEACHER_OBS_GROUPS = {"actor": ("policy", "privileged"), "critic": ("policy", "privileged")}


def unitree_aliengo_direction_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
    """Create PPO runner configuration for Unitree AlienGo direction task."""
    return RslRlOnPolicyRunnerCfg(
        obs_groups=dict(_TEACHER_OBS_GROUPS),
        actor=RslRlModelCfg(
            hidden_dims=(512, 256, 256, 128),
            activation="elu",
            obs_normalization=True,
            distribution_cfg={
                "class_name": "GaussianDistribution",
                "init_std": 1.0,
                "std_type": "scalar",
            },
        ),
        critic=RslRlModelCfg(
            hidden_dims=(512, 256, 256, 128),
            activation="elu",
            obs_normalization=True,
        ),
        algorithm=RslRlSymmetricPpoAlgorithmCfg(
            num_learning_epochs=5,
            num_mini_batches=4,
            learning_rate=1.0e-3,
            schedule="adaptive",
            gamma=0.99,
            lam=0.95,
            entropy_coef=0.005,
            desired_kl=0.01,
            max_grad_norm=1.0,
            value_loss_coef=1.0,
            use_clipped_value_loss=True,
            clip_param=0.2,
            symmetry_cfg=RslRlSymmetryCfg(
                use_data_augmentation=True,
                data_augmentation_func=compute_symmetric_states,
            ),
        ),
        experiment_name="aliengo_direction",
        logger="tensorboard",
        save_interval=50,
        num_steps_per_env=24,
        max_iterations=1500,
    )


def unitree_aliengo_direction_distillation_runner_cfg() -> RslRlDistillationRunnerCfg:
    """Create DAgger distillation runner configuration for Unitree AlienGo direction task."""
    return RslRlDistillationRunnerCfg(
        obs_groups={"student": ("policy",), "teacher": ("policy", "privileged")},
        student=RslRlModelCfg(
            class_name="RNNModel",
            hidden_dims=(512, 256, 256, 128),
            activation="elu",
            obs_normalization=True,
            distribution_cfg={
                "class_name": "GaussianDistribution",
                "init_std": 0.1,
                "std_type": "scalar",
            },
            rnn_type="gru",
            rnn_hidden_dim=256,
            rnn_num_layers=1,
        ),
        teacher=RslRlModelCfg(
            hidden_dims=(512, 256, 256, 128),
            activation="elu",
            obs_normalization=True,
            distribution_cfg={
                "class_name": "GaussianDistribution",
                "init_std": 0.0,
                "std_type": "scalar",
            },
        ),
        algorithm=RslRlDistillationAlgorithmCfg(
            num_learning_epochs=2,
            learning_rate=1.0e-3,
            gradient_length=15,
            loss_type="mse",
        ),
        experiment_name="aliengo_direction",
        logger="tensorboard",
        save_interval=50,
        num_steps_per_env=24,
        max_iterations=1000,
    )
