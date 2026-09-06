from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal

from mjlab.rl import RslRlBaseRunnerCfg, RslRlModelCfg, RslRlPpoAlgorithmCfg


@dataclass
class RslRlSymmetryCfg:
    """Config for ``rsl_rl.extensions.Symmetry`` data augmentation / mirror loss."""

    data_augmentation_func: Callable[..., Any]
    """Callable that generates mirrored observations/actions. 
    Signature: ``(env, obs, actions) -> (obs_aug, actions_aug)``."""
    use_data_augmentation: bool = False
    """Whether to append mirrored samples to every mini-batch."""
    use_mirror_loss: bool = False
    """Whether to add an auxiliary mirror loss term to the policy loss."""
    mirror_loss_coeff: float = 0.0
    """Scaling factor for the mirror loss when ``use_mirror_loss`` is True."""


@dataclass
class RslRlSymmetricPpoAlgorithmCfg(RslRlPpoAlgorithmCfg):
    """``RslRlPpoAlgorithmCfg`` with symmetry-augmented data augmentation."""

    symmetry_cfg: RslRlSymmetryCfg | None = None
    """Symmetry data-augmentation config. ``rsl_rl`` injects the running env 
    into it at construction time, so this only needs the augmentation function/flags."""


@dataclass
class RslRlDistillationAlgorithmCfg:
    """Config for the ``rsl_rl.algorithms.Distillation`` (DAgger) algorithm."""

    num_learning_epochs: int = 1
    """The number of learning epochs per update."""
    gradient_length: int = 15
    """Number of steps to backpropagate gradients through (BPTT truncation)."""
    learning_rate: float = 1e-3
    """The learning rate."""
    max_grad_norm: float | None = None
    """The maximum gradient norm for the student."""
    loss_type: Literal["mse", "huber"] = "mse"
    """The distillation loss function."""
    optimizer: Literal["adam", "adamw", "sgd", "rmsprop"] = "adam"
    """The optimizer to use."""
    class_name: str = "Distillation"
    """Algorithm class name resolved by RSL-RL."""


@dataclass
class RslRlDistillationRunnerCfg(RslRlBaseRunnerCfg):
    """Config for ``rsl_rl.runners.DistillationRunner`` (teacher -> student DAgger)."""

    class_name: str = "DistillationRunner"
    """The runner class name."""
    obs_groups: dict[str, tuple[str, ...]] = field(
        default_factory=lambda: {"student": ("student",), "teacher": ("teacher",)},
    )
    student: RslRlModelCfg = field(
        default_factory=lambda: RslRlModelCfg(
            distribution_cfg={
                "class_name": "GaussianDistribution",
                "init_std": 1.0,
                "std_type": "scalar",
            }
        )
    )
    """The student ("deployable") model configuration."""
    teacher: RslRlModelCfg = field(default_factory=RslRlModelCfg)
    """The teacher (privileged) model configuration."""
    algorithm: RslRlDistillationAlgorithmCfg = field(default_factory=RslRlDistillationAlgorithmCfg)
    """The algorithm configuration."""
