"""Train TD3/SAC residual actors while keeping the DAgger network frozen."""
import argparse
import hashlib
import json
import platform
from pathlib import Path
import time

import numpy as np
import torch
from stable_baselines3 import SAC, TD3
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, CallbackList
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.env_checker import check_env

from .residual_env import ResidualTrackingEnv, read_config
from .rl_backend import workspace_root, sha256
from .residual_exploration import WarmupSAC
from .residual_validation import PairedValidationCallback


def tensor_hash(model):
    digest = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


class ValidationCallback(BaseCallback):
    """An independent physics process prevents evaluation from resetting training."""
    def __init__(self, env, output, frequency):
        super().__init__()
        self.env = env
        self.output = Path(output)
        self.frequency = int(frequency)
        self.best = -float("inf")

    def _on_step(self):
        if not self.frequency or self.num_timesteps % self.frequency:
            return True
        scores, success = [], []
        for i, route in enumerate(self.env.config["routes"]):
            obs, _ = self.env.reset(seed=10000+i, options={"route": route, "randomize": True})
            done, total = False, 0.0
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = self.env.step(action)
                total += reward
                done = terminated or truncated
            scores.append(total)
            success.append(info["is_success"])
        score = float(np.mean(scores))
        row = dict(timesteps=self.num_timesteps, mean_return=score,
                   success_rate=float(np.mean(success)), route_returns=scores)
        with (self.output/"validation.jsonl").open("a") as stream:
            stream.write(json.dumps(row)+"\n")
        self.logger.record("validation/return", score)
        self.logger.record("validation/success_rate", float(np.mean(success)))
        if score>self.best:
            self.best = score
            self.model.save(self.output/"best")
        return True


def build_model(algo, env, config, seed, device, tensorboard_log=None):
    t = config["training"]
    shared = dict(learning_rate=t["learning_rate"], buffer_size=t["buffer_size"],
                  learning_starts=t["learning_starts"], batch_size=t["batch_size"],
                  gamma=t["gamma"], tau=t["tau"], train_freq=1, gradient_steps=1,
                  policy_kwargs=dict(net_arch=t["net_arch"]), seed=seed, device=device,
                  tensorboard_log=tensorboard_log, verbose=1)
    if algo=="td3":
        noise = NormalActionNoise(np.zeros(3), np.full(3,t["td3_noise_std"]))
        model = TD3("MlpPolicy", env, action_noise=noise, **shared)
        # Deterministic actor starts at exactly zero residual; exploration
        # noise and warmup still operate within the configured residual cap.
        torch.nn.init.zeros_(model.actor.mu[-2].weight)
        torch.nn.init.zeros_(model.actor.mu[-2].bias)
        model.actor_target.load_state_dict(model.actor.state_dict())
    else:
        model_class = WarmupSAC if "warmup_mode" in t else SAC
        model = model_class("MlpPolicy", env, ent_coef=t["sac_ent_coef"], **shared)
        if isinstance(model, WarmupSAC):
            model.configure_residual_warmup(t)
        torch.nn.init.zeros_(model.actor.mu.weight)
        torch.nn.init.zeros_(model.actor.mu.bias)
        torch.nn.init.zeros_(model.actor.log_std.weight)
        torch.nn.init.constant_(model.actor.log_std.bias, -2.0)
    return model


def train(args):
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested but is unavailable in this process; "
            "check Docker GPU passthrough or use --device cpu")
    config = read_config(args.config)
    if args.learning_starts is not None:
        config["training"]["learning_starts"] = args.learning_starts
    if args.batch_size is not None:
        config["training"]["batch_size"] = args.batch_size
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output/"config.json").write_text(json.dumps(config,indent=2)+"\n")
    gpu = torch.cuda.get_device_name(0) if device == "cuda" else None
    manifest = dict(algo=args.algo, seed=args.seed, total_timesteps=args.total_timesteps,
                    config=config, torch=torch.__version__, python=platform.python_version(),
                    requested_device=args.device, device=device, gpu_name=gpu,
                    backend="stonefish_fixed_step", residual_action="bounded body Fx,Fy,Nz",
                    validation_seeds=[10000+i for i in range(len(config["routes"]))])
    (output/"training.json").write_text(json.dumps(manifest,indent=2)+"\n")
    env, validation = None, None
    model = None
    try:
        raw = ResidualTrackingEnv(config, output/"train_env", training=True,
                                  record_steps=args.record_train_steps)
        env = Monitor(raw, str(output/"monitor.csv"), info_keywords=("is_success",))
        frozen = tensor_hash(raw.policy.model)
        print("Training %s seed=%d steps=%d on %s%s" %
              (args.algo, args.seed, args.total_timesteps, device,
               " (" + gpu + ")" if gpu else ""), flush=True)
        model = build_model(args.algo, env, config, args.seed, device,
                             str(output/"tensorboard"))
        callbacks = []
        if args.eval_freq:
            validation = ResidualTrackingEnv(config, output/"validation_env", training=False)
            if "validation" in config:
                callbacks.append(PairedValidationCallback(validation, output, args.eval_freq, config["validation"]))
            else:
                callbacks.append(ValidationCallback(validation, output, args.eval_freq))
        if args.checkpoint_freq:
            callbacks.append(CheckpointCallback(args.checkpoint_freq, str(output/"checkpoints"),
                                                 name_prefix=args.algo,
                                                 save_replay_buffer=args.save_replay_buffer))
        model.learn(total_timesteps=args.total_timesteps,
                    callback=CallbackList(callbacks), tb_log_name=args.algo)
        model.save(output/"final")
        if args.save_replay_buffer:
            model.save_replay_buffer(output/"replay_buffer.pkl")
        if frozen!=tensor_hash(raw.policy.model):
            raise AssertionError("frozen baseline changed during RL training")
        (output/"completed.json").write_text(json.dumps(dict(
            timesteps=model.num_timesteps, updates=model._n_updates,
            device=device, gpu_name=gpu,
            baseline_tensor_sha256=frozen, baseline_unchanged=True,
            actor_updates=(model._n_updates // model.policy_delay if args.algo == "td3" else model._n_updates),
            warmup_mode=config["training"].get("warmup_mode", "uniform"),
            note="best.zip exists only after validation; final.zip is the final iterate; ablations also write selection.json"),indent=2)+"\n")
        print("Training artifacts: %s" % output, flush=True)
    finally:
        if env is not None:
            env.close()
        if validation is not None:
            validation.close()


def smoke(args):
    config = read_config(args.config)
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    env = ResidualTrackingEnv(config,output/"env",training=False)
    try:
        check_env(env, warn=True)
        frozen = tensor_hash(env.policy.model)
        replay = []
        for repeat in range(2):
            obs, _ = env.reset(seed=0,options={"route":"right_angle","randomize":False})
            trace = []
            for i in range(args.steps):
                obs,reward,terminated,truncated,info = env.step(np.zeros(3,dtype=np.float32))
                trace.append([env.state[k] for k in ("x","y","yaw","u","v","r")])
                if terminated or truncated:
                    break
            replay.append(np.asarray(trace))
        np.testing.assert_allclose(replay[0],replay[1],rtol=0,atol=1e-6)
        assert frozen==tensor_hash(env.policy.model)
        assert np.max(np.abs(env.previous_residual))==0
        # Pausing the learner for wall time must not advance the simulator.
        stamp = env.state["stamp"]
        time.sleep(0.2)
        state = env.backend.step(np.zeros(6),env.dt)
        assert abs(state["stamp"]-stamp-env.dt)<1e-6
        report=dict(backend="real_stonefish", gym_check=True, zero_residual_repeatable=True,
                    baseline_unchanged=True, fixed_step=True, tested_steps=len(replay[0]),
                    ros_reproduction="requires separate ROS evaluation; this checks native repeatability")
        (output/"smoke.json").write_text(json.dumps(report,indent=2)+"\n")
        print(json.dumps(report,indent=2),flush=True)
    finally:
        env.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest="command",required=True)
    p=sub.add_parser("smoke",help="check Gym, fixed-step physics and frozen-baseline repeatability")
    p.add_argument("--config")
    p.add_argument("--output",required=True)
    p.add_argument("--steps",type=int,default=50)
    p=sub.add_parser("train",help="train TD3 or SAC on the same residual environment")
    p.add_argument("--config")
    p.add_argument("--output",required=True)
    p.add_argument("--algo",choices=("td3","sac"),required=True)
    p.add_argument("--seed",type=int,default=0)
    p.add_argument("--total-timesteps",type=int,default=100000)
    p.add_argument("--device",choices=("auto","cpu","cuda"),default="auto")
    p.add_argument("--eval-freq",type=int,default=5000)
    p.add_argument("--checkpoint-freq",type=int,default=5000)
    p.add_argument("--learning-starts",type=int)
    p.add_argument("--batch-size",type=int)
    p.add_argument("--record-train-steps",action="store_true")
    p.add_argument("--save-replay-buffer",action="store_true")
    args=parser.parse_args()
    if args.command=="smoke":
        if args.steps<1: parser.error("steps must be positive")
        smoke(args)
    else:
        if args.total_timesteps<1 or args.eval_freq<0 or args.checkpoint_freq<0:
            parser.error("invalid training budget or callback interval")
        train(args)


if __name__=="__main__":
    main()
