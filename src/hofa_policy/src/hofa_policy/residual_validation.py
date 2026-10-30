"""Paired validation for reward/warmup ablations, without test-set selection."""
import json
from pathlib import Path
import numpy as np
import torch
from stable_baselines3.common.callbacks import BaseCallback
from .residual_selection import aggregate_metrics, assess_candidate, candidate_key


class PairedValidationCallback(BaseCallback):
    def __init__(self, env, output, frequency, protocol):
        super().__init__()
        self.env, self.output = env, Path(output)
        self.frequency, self.protocol = int(frequency), protocol
        self.best_key = None
        self.baseline = None

    def rollout(self, zero=False):
        metrics, policy_log_std = [], []
        for case in self.protocol["cases"]:
            obs, _ = self.env.reset(seed=case["seed"], options={
                "route": case["route"], "randomize": case["randomized"]})
            done, steps = False, 0
            while not done:
                if zero:
                    action = np.zeros(3, dtype=np.float32)
                else:
                    action, _ = self.model.predict(obs, deterministic=True)
                    if steps % 20 == 0 and hasattr(self.model.actor, "get_action_dist_params"):
                        with torch.no_grad():
                            obs_tensor, _ = self.model.policy.obs_to_tensor(obs)
                            _, log_std, _ = self.model.actor.get_action_dist_params(obs_tensor)
                            policy_log_std.append(log_std.detach().cpu().numpy().ravel())
                obs, _, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                steps += 1
            metrics.append(dict(case=case["label"], **info["episode_metrics"]))
        return aggregate_metrics(metrics), metrics, policy_log_std

    def _on_training_start(self):
        self.baseline, episodes, _ = self.rollout(zero=True)
        (self.output / "zero_residual_validation.json").write_text(json.dumps(dict(
            metrics=self.baseline, cases=self.protocol["cases"], episodes=episodes), indent=2) + "\n")
        self.write_selection(None, None)

    def write_selection(self, candidate, assessment):
        selection = dict(
            criterion="minimum cross-track RMSE subject to common physical guards; all validation cases included",
            rules=self.protocol["selection"], baseline=self.baseline, best_model=candidate,
            assessment=assessment,
            selected="best.zip" if assessment and assessment["preferred_to_zero"] else "zero_residual",
            note="best.zip is a diagnostic model candidate even when zero residual is selected",
        )
        (self.output / "selection.json").write_text(json.dumps(selection, indent=2) + "\n")

    def _on_step(self):
        if not self.frequency or self.num_timesteps % self.frequency:
            return True
        metrics, episodes, log_std = self.rollout()
        assessment = assess_candidate(metrics, self.baseline, self.protocol["selection"])
        row = dict(timesteps=self.num_timesteps, mean_return=metrics["training_return"],
                   success_rate=metrics["success_rate"], comparison_return=metrics["comparison_return"],
                   metrics=metrics, assessment=assessment, episodes=episodes)
        if getattr(self.model, "log_ent_coef", None) is not None:
            row["ent_coef"] = float(self.model.log_ent_coef.detach().exp().cpu())
        if log_std:
            row["policy_mean_log_std_per_axis"] = np.mean(log_std, axis=0).tolist()
        with (self.output / "validation.jsonl").open("a") as stream:
            stream.write(json.dumps(row) + "\n")
        self.logger.record("validation/return", metrics["training_return"])
        self.logger.record("validation/comparison_return", metrics["comparison_return"])
        self.logger.record("validation/cross_track_rmse_m", metrics["cross_track_rmse_m"])
        self.logger.record("validation/success_rate", metrics["success_rate"])
        key = candidate_key(metrics, assessment)
        if self.best_key is None or key > self.best_key:
            self.best_key = key
            self.model.save(self.output / "best")
            self.write_selection(dict(timesteps=self.num_timesteps, **metrics), assessment)
        return True
