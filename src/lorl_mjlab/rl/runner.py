import torch
from mjlab.rl import RslRlVecEnvWrapper
from rsl_rl.env import VecEnv
from rsl_rl.runners import DistillationRunner


class LorlDistillationRunner(DistillationRunner):
    """Distillation runner with mjlab's checkpoint/env-state persistence."""

    env: RslRlVecEnvWrapper

    def __init__(self, env: VecEnv, train_cfg: dict, log_dir: str | None = None, device: str = "cpu") -> None:
        # Strip None-valued optional configs so MLPModel doesn't receive them
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
        """Save checkpoint.

        Extends the base implementation to persist the environment's
        common_step_counter and to respect the ``upload_model`` config flag.
        """
        env_state = {"common_step_counter": self.env.unwrapped.common_step_counter}
        infos = {**(infos or {}), "env_state": env_state}
        saved_dict = self.alg.save()
        saved_dict["iter"] = self.current_learning_iteration
        saved_dict["infos"] = infos
        torch.save(saved_dict, path)
        if self.cfg["upload_model"]:
            self.logger.save_model(path, self.current_learning_iteration)

    @staticmethod
    def _translate_load_cfg(load_cfg: dict | None) -> dict | None:
        """Map PPO's ``"actor"`` selector onto distillation's ``"student"``.

        ``mjlab.scripts.play`` asks every runner for ``{"actor": True}``, but
        ``Distillation.load`` only understands ``student``/``teacher``/``optimizer``/
        ``iteration``. Because the dict is non-``None`` it also skips its own defaults,
        so the unmapped request silently loads *nothing* and play runs a freshly
        initialized student -- a robot that stands there doing nothing.
        """
        if load_cfg is None or "actor" not in load_cfg:
            return load_cfg
        translated = {k: v for k, v in load_cfg.items() if k != "actor"}
        translated["student"] = load_cfg["actor"]
        return translated

    def load(
        self, path: str, load_cfg: dict | None = None, strict: bool = True, map_location: str | None = None
    ) -> dict:
        """Load checkpoint.

        Extends the base implementation to restore
        common_step_counter and current_learning_iteration to preserve curricula state.
        """
        loaded_dict = torch.load(path, map_location=map_location, weights_only=False)
        load_iteration = self.alg.load(loaded_dict, self._translate_load_cfg(load_cfg), strict)
        if load_iteration:
            self.current_learning_iteration = loaded_dict["iter"]
        infos = loaded_dict["infos"]
        if infos and "env_state" in infos:
            self.env.unwrapped.common_step_counter = infos["env_state"]["common_step_counter"]
        return infos
