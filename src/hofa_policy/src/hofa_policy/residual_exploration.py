"""Local warmup for SAC, with unchanged sampling after learning_starts."""
import numpy as np
from stable_baselines3 import SAC


class WarmupSAC(SAC):
    def configure_residual_warmup(self, config):
        self.residual_warmup_mode = config.get("warmup_mode", "uniform")
        self.residual_warmup_std = float(config.get("warmup_std", 0.05))
        self.residual_warmup_clip = float(config.get("warmup_clip", 0.15))
        if self.residual_warmup_mode not in ("uniform", "small_gaussian"):
            raise ValueError("unknown residual warmup mode")
        if not 0 < self.residual_warmup_std or not 0 < self.residual_warmup_clip <= 1:
            raise ValueError("positive warmup std and clip in (0, 1] required")
        self.residual_warmup_rng = np.random.default_rng(self.seed)
        if not (np.all(self.action_space.low == -1) and np.all(self.action_space.high == 1)):
            raise ValueError("residual warmup requires normalized [-1, 1] actions")

    def _sample_action(self, learning_starts, action_noise=None, n_envs=1):
        if (getattr(self, "residual_warmup_mode", "uniform") != "small_gaussian"
                or self.num_timesteps >= learning_starts):
            return super()._sample_action(learning_starts, action_noise, n_envs)
        if action_noise is not None or self.use_sde:
            raise ValueError("local warmup is defined for ordinary SAC without extra action noise")
        scaled = np.clip(self.residual_warmup_rng.normal(
            0, self.residual_warmup_std, size=(n_envs, *self.action_space.shape)),
            -self.residual_warmup_clip, self.residual_warmup_clip).astype(np.float32)
        # SB3 records normalized actions. The environment receives the matching
        # unscaled action; the env's deterministic slew limiter stays in its MDP.
        return scaled.copy(), scaled.copy()
