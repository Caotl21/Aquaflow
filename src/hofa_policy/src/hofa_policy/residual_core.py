"""Shared observation/action contract for Gym training and ROS deployment."""
import json
from pathlib import Path
import numpy as np
from .rl_backend import sha256

OBSERVATION_SIZE = 212


def observation_size(config):
    return OBSERVATION_SIZE + (3 if config.get("observation", {}).get(
        "include_previous_desired_residual", False) else 0)


def observation(state, reference, previous_wrench, base, previous_residual, model, limits,
                previous_desired=None):
    result=np.r_[state/model.state_scale.cpu().numpy(),
                 (reference/model.reference_scale.cpu().numpy()).ravel(),
                 previous_wrench/np.array([10,5,1]), base/np.array([10,5,1]),
                 previous_residual/limits].astype(np.float32)
    size = OBSERVATION_SIZE
    if previous_desired is not None:
        previous_desired = np.asarray(previous_desired, dtype=float)
        if previous_desired.shape != (3,):
            raise ValueError("previous desired residual must have three components")
        result = np.r_[result, previous_desired / limits].astype(np.float32)
        size += 3
    if result.shape!=(size,) or not np.isfinite(result).all():
        raise ValueError("invalid residual policy observation")
    return result


def compose(base, action, previous, limits, slew):
    action=np.asarray(action,dtype=float)
    if action.shape!=(3,) or not np.isfinite(action).all():
        raise ValueError("residual action must contain three finite values")
    desired=np.clip(action,-1,1)*limits
    residual=np.clip(desired,previous-slew,previous+slew)
    return np.asarray(base)+residual,residual


class ResidualActor:
    def __init__(self, model_path, baseline_path, algo, baseline_model):
        from stable_baselines3 import SAC,TD3
        path=Path(model_path).resolve()
        parent=path.parent
        if not (parent/"training.json").exists(): parent=parent.parent
        manifest=json.loads((parent/"training.json").read_text())
        provenance=json.loads((parent/"train_env/provenance.json").read_text())
        if manifest["algo"]!=algo or sha256(baseline_path)!=provenance["baseline_sha256"]:
            raise ValueError("residual algorithm or baseline checkpoint mismatch")
        self.config=manifest["config"]
        self.limits=np.asarray(self.config["residual_limit"])
        self.slew=np.asarray(self.config["residual_slew_per_step"])
        self.previous=np.zeros(3)
        self.previous_desired = np.zeros(3)
        self.include_desired = observation_size(self.config) != OBSERVATION_SIZE
        self.last_diagnostics = None
        self.baseline_model=baseline_model
        self.model=(TD3 if algo=="td3" else SAC).load(path,device="cpu")
        if self.model.observation_space.shape!=(observation_size(self.config),):
            raise ValueError("residual model observation contract mismatch")

    def predict(self,state,reference,previous_wrench,base):
        obs=observation(state,reference,previous_wrench,base,self.previous,
                        self.baseline_model,self.limits,
                        self.previous_desired if self.include_desired else None)
        action,_=self.model.predict(obs,deterministic=True)
        previous = self.previous.copy()
        command,self.previous=compose(base,action,self.previous,self.limits,self.slew)
        self.last_diagnostics = {
            "action": np.asarray(action).copy(),
            "desired": np.clip(action, -1, 1) * self.limits,
            "previous_residual": previous,
            "previous_desired": self.previous_desired.copy(),
            "applied": self.previous.copy(),
            "composed": np.asarray(command).copy(),
        }
        self.previous_desired = self.last_diagnostics["desired"].copy()
        return command

    def reset(self):
        self.previous.fill(0)
        self.previous_desired.fill(0)
        self.last_diagnostics = None
