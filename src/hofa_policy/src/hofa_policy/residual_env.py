"""Gymnasium residual control on real, synchronously stepped Stonefish."""
import copy
import csv
import json
from pathlib import Path
import gymnasium as gym
from gymnasium import spaces
import numpy as np

from .features import reference_features, state_features
from .shadow_core import ShadowPolicy
from .rl_backend import StonefishBackend, flatten_scenario, sha256, workspace_root
from .rl_actuation import Actuation, DepthPID
from .rl_reference import Reference, make_route, ROUTES
from .residual_core import OBSERVATION_SIZE, observation_size, observation, compose
from .residual_reward import potential_values, reward_terms


def read_config(path=None):
    root = workspace_root()
    path = Path(path or root/"src/hofa_policy/config/residual_rl.json")
    config = json.loads(path.read_text())
    if config.get("schema") != "hofa_residual_rl_v1":
        raise ValueError("residual RL config schema mismatch")
    return config


def write_csv(path, rows):
    if not rows:
        return
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


class ResidualTrackingEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, config=None, output_dir=None, training=True,
                 record_steps=False, device="cpu", backend=None, policy=None):
        super().__init__()
        self.config = copy.deepcopy(config or read_config())
        self.root = workspace_root()
        self.output = Path(output_dir or self.root/"artifacts/hofa_rl_env").resolve()
        self.output.mkdir(parents=True, exist_ok=True)
        self.training = bool(training)
        self.record_steps = bool(record_steps)
        self.dt = float(self.config["control_dt_s"])
        if abs(self.dt-0.1)>1e-9:
            raise ValueError("frozen DAgger contract requires a 10 Hz controller")
        self.limits = np.asarray(self.config["residual_limit"], dtype=float)
        self.slew = np.asarray(self.config["residual_slew_per_step"], dtype=float)
        if self.limits.shape!=(3,) or np.any(self.limits<=0) or self.slew.shape!=(3,) or np.any(self.slew<=0):
            raise ValueError("positive per-axis residual and slew limits required")
        self.action_space = spaces.Box(-1.0, 1.0, shape=(3,), dtype=np.float32)
        # state(3), horizon(20*10), previous commanded wrench(3), base(3),
        # previous applied residual(3). All are available in ROS deployment.
        self.include_desired = observation_size(self.config) != OBSERVATION_SIZE
        if self.config["reward"].get("desired_smooth", 0) and not self.include_desired:
            raise ValueError("desired smoothing requires previous desired residual in observation")
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(observation_size(self.config),), dtype=np.float32)
        self.policy = policy or ShadowPolicy(str(self.path("baseline_checkpoint")), device, 1)
        self.policy.model.requires_grad_(False)
        self.policy.model.eval()
        self.actuation = Actuation(self.root)
        self.episode = 0
        self.active = False
        self.rows = []
        self.backend = backend
        if backend is None:
            snapshot = self.output/"scenario.scn"
            sources = flatten_scenario(self.path("scenario"), snapshot, self.root)
            manifest = dict(schema="hofa_residual_env_provenance_v1", backend="stonefish_fixed_step",
                            physical_scenario_sources=sources, flattened_scenario_sha256=sha256(snapshot),
                            baseline_sha256=sha256(self.path("baseline_checkpoint")),
                            bridge_sha256=sha256(self.path("bridge_executable")),
                            config=self.config)
            code = ["rl_reference.py", "rl_actuation.py", "residual_env.py", "rl_backend.py",
                    "features.py", "shadow_core.py", "model.py", "contract.py", "residual_core.py",
                    "residual_reward.py"]
            manifest["code_sha256"] = {name: sha256(Path(__file__).parent/name) for name in code}
            for relative in ["src/hofa_mpc_ros/src/hofa_mpc_ros/arc_profile.py",
                             "src/aquaflow_stonefish/scripts/brics6_thruster_allocator.py"]:
                manifest["code_sha256"][relative] = sha256(self.root/relative)
            (self.output/"provenance.json").write_text(json.dumps(manifest, indent=2)+"\n")
            self.backend = StonefishBackend(self.path("bridge_executable"), self.path("data_dir"),
                                             snapshot, self.config["physics_hz"], self.output)

    def path(self, key):
        path = Path(self.config[key]).expanduser()
        return path if path.is_absolute() else self.root/path

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if self.active:
            self._finish("reset_interrupted")
        options = options or {}
        route_name = options.get("route") or str(self.np_random.choice(self.config["routes"]))
        if route_name not in ROUTES:
            raise ValueError("unknown route: %s" % route_name)
        randomize = options.get("randomize", self.training and self.config["randomize_train_routes"])
        route = make_route(route_name, self.config["goal_depth"], self.np_random, randomize)
        self.reference = Reference(route, self.config["profile"])
        self.depth = DepthPID(self.config["goal_depth"])
        start = route["points"][0]
        self.state = self.backend.reset([start[0], start[1], start[2], start[3]])
        # Hold depth at the start while clearing the small IC transient.
        for _ in range(round(self.config["settle_s"]/self.dt)):
            pwm, _, _, _ = self.actuation.allocate(np.zeros(3), self.depth.command(self.state, self.dt))
            self.state = self.backend.step(pwm, self.dt)
        self.start_stamp = self.state["stamp"]
        self.elapsed, self.steps = 0.0, 0
        self.previous_wrench = np.zeros(3, dtype=np.float32)
        self.previous_residual = np.zeros(3)
        self.previous_desired = np.zeros(3)
        self.reward_total = 0.0
        self.episode += 1
        self.rows, self.odom_rows, self.force_rows, self.pwm_rows = [], [], [], []
        self.status_rows = []
        self.window_rows = []
        self.reference.project(self.state["x"], self.state["y"], update=True)
        start_ref = self.reference.sample([self.reference.arc])
        start_yaw_error = float(np.arctan2(
            np.sin(self.state["yaw"] - start_ref["yaw"][0]),
            np.cos(self.state["yaw"] - start_ref["yaw"][0])))
        start_rate_error = float(self.state["r"] - start_ref["dpsi"][0])
        start_terminal_error = float(np.arctan2(
            np.sin(self.state["yaw"] - self.reference.profile["yaw"][-1]),
            np.cos(self.state["yaw"] - self.reference.profile["yaw"][-1])))
        self.previous_potential = potential_values(
            self.config, float(start_ref["dpsi"][0]), start_rate_error,
            start_terminal_error,
            float(self.reference.profile["total_length"] - self.reference.arc))
        self.active = True
        return self._observation(), {"route": route_name, "backend": "stonefish_fixed_step"}

    def _observation(self):
        self.window = self.reference.window(self.elapsed)
        state = state_features(self.state)
        features = reference_features(self.state, self.window)
        self.base, _, _ = self.policy.predict(state, features, self.previous_wrench)
        model = self.policy.model
        obs = observation(state,features,self.previous_wrench,self.base,
                          self.previous_residual,model,self.limits,
                          self.previous_desired if self.include_desired else None)
        if obs.shape != self.observation_space.shape or not np.isfinite(obs).all():
            raise RuntimeError("nonfinite or invalid RL observation")
        return obs

    def step(self, action):
        if not self.active:
            raise RuntimeError("call reset before step")
        action = np.asarray(action, dtype=float)
        if action.shape!=(3,) or not np.isfinite(action).all():
            raise ValueError("action must contain three finite values")
        base = self.base.copy()
        desired = np.clip(action, -1, 1) * self.limits
        previous_desired = self.previous_desired.copy()
        previous_residual = self.previous_residual.copy()
        command,residual = compose(base,action,self.previous_residual,self.limits,self.slew)
        if self.record_steps:
            self.window_rows.extend(dict(stamp=self.state["stamp"], index=i, **point)
                                    for i,point in enumerate(self.window))
        pwm, realized, forces, saturated = self.actuation.allocate(
            command, self.depth.command(self.state, self.dt))
        old_arc = self.reference.arc
        old_stamp = self.state["stamp"]
        self.state = self.backend.step(pwm, self.dt)
        if abs(self.state["stamp"]-old_stamp-self.dt)>1e-6:
            raise RuntimeError("Stonefish physics time did not advance by one control interval")
        self.steps += 1
        self.elapsed = self.state["stamp"]-self.start_stamp
        arc, cross, distance = self.reference.project(self.state["x"], self.state["y"], update=True)
        ref = self.reference.sample([arc])
        yaw_error = float(np.arctan2(np.sin(self.state["yaw"]-ref["yaw"][0]),
                                    np.cos(self.state["yaw"]-ref["yaw"][0])))
        rate_error = float(self.state["r"]-ref["dpsi"][0])
        terminal_yaw_error = float(np.arctan2(
            np.sin(self.state["yaw"] - self.reference.profile["yaw"][-1]),
            np.cos(self.state["yaw"] - self.reference.profile["yaw"][-1])))
        scheduled = float(np.interp(self.elapsed, self.reference.profile["time"],
                                     self.reference.profile["s"]))
        lag = scheduled-arc
        goal = self.reference.route["goal"]
        final_distance = float(np.hypot(self.state["x"]-goal[0], self.state["y"]-goal[1]))
        remaining_arc = float(self.reference.profile["total_length"] - arc)
        current_potential = potential_values(
            self.config, float(ref["dpsi"][0]), rate_error, terminal_yaw_error,
            remaining_arc)
        success = (self.elapsed>2 and final_distance<0.25 and
                   self.state["speed_world"]<0.08 and
                   arc>self.reference.profile["total_length"]-0.3)
        failure = (abs(self.state["x"])>6.6 or abs(self.state["y"])>3.6 or
                   not 0.2<self.state["z"]<3.6 or self.state["speed_world"]>1.0 or
                   abs(self.state["r"])>1.2 or distance>0.7)
        truncated = self.elapsed>=self.config["episode_timeout_s"]
        reward_inputs = (self.dt, arc-old_arc, cross, yaw_error, rate_error, lag,
                         residual, self.previous_residual, self.limits, self.slew, success, failure)
        terms = reward_terms(
            self.config, *reward_inputs, reference_yaw_rate=float(ref["dpsi"][0]),
            remaining_distance_m=final_distance, remaining_arc_m=remaining_arc,
            terminal_yaw_error=terminal_yaw_error,
            potential_previous=self.previous_potential,
            potential_current=current_potential, desired_residual=desired,
            previous_desired_residual=previous_desired)
        common_terms = reward_terms(
            self.config, *reward_inputs, comparison=True,
            reference_yaw_rate=float(ref["dpsi"][0]),
            remaining_distance_m=final_distance, remaining_arc_m=remaining_arc,
            terminal_yaw_error=terminal_yaw_error,
            potential_previous=self.previous_potential,
            potential_current=current_potential, desired_residual=desired,
            previous_desired_residual=previous_desired)
        reward = float(sum(terms.values()))
        self.reward_total += reward
        # The frozen model sees the published combined wrench, matching the
        # existing ROS policy's command-history convention. The reconstructed
        # allocator force is logged separately for clipping/deadzone analysis.
        self.previous_wrench = command.astype(np.float32)
        self.previous_residual = residual.copy()
        self.previous_desired = desired.copy()
        self.previous_potential = current_potential
        row = dict(t_rel=self.elapsed, arc_m=arc, cross_track_m=cross,
                   yaw_error_deg=float(np.degrees(yaw_error)), yaw_rate_error=rate_error,
                   terminal_yaw_error_deg=float(np.degrees(terminal_yaw_error)),
                   remaining_arc_m=remaining_arc,
                   reference_yaw_rate=float(ref["dpsi"][0]), lag_m=lag,
                   reward=reward, saturated=int(saturated))
        row.update({"reward_"+key: float(value) for key, value in terms.items()})
        row["comparison_reward"] = float(sum(common_terms.values()))
        row.update({"comparison_reward_"+key: float(value) for key, value in common_terms.items()})
        for i, key in enumerate(("fx", "fy", "nz")):
            row.update({"base_"+key: float(base[i]), "residual_"+key: float(residual[i]),
                        "command_"+key: float(command[i]), "realized_"+key: float(realized[i]),
                        "action_"+key: float(action[i]),
                        "desired_"+key: float(desired[i]),
                        "previous_desired_"+key: float(previous_desired[i]),
                        "desired_change_"+key: float(desired[i]-previous_desired[i]),
                        "residual_change_"+key: float(residual[i]-previous_residual[i]),
                        "slew_limited_"+key: int(abs(desired[i]-residual[i])>1e-7)})
        self.rows.append(row)
        self.odom_rows.append(dict(self.state, t_rel=self.elapsed))
        self.force_rows.append(dict(stamp=self.state["stamp"], fx=float(command[0]),
                                    fy=float(command[1]), nz=float(command[2])))
        self.pwm_rows.append(dict(stamp=self.state["stamp"],
                                  **{"pwm_%d" % i: float(v) for i, v in enumerate(pwm)}))
        info = dict(row, is_success=bool(success), final_distance_to_goal_m=final_distance,
                    backend="stonefish_fixed_step")
        observation = self._observation()
        if success or failure or truncated:
            reason = "goal_reached" if success and not failure else "safety_stop" if failure else "timeout"
            metrics = self._finish(reason)
            info["episode_metrics"] = metrics
        return observation, reward, bool(success or failure), bool(truncated and not(success or failure)), info

    def _finish(self, reason):
        self.active = False
        rows = self.rows
        def stats(key, turns=False):
            values = np.asarray([row[key] for row in rows
                                 if not turns or abs(row["reference_yaw_rate"])>=0.05])
            if not len(values):
                return None
            return dict(rmse=float(np.sqrt(np.mean(values**2))),
                        max_abs=float(np.max(np.abs(values))))
        p = self.reference.profile
        goal = self.reference.route["goal"]
        metrics = dict(backend="stonefish_fixed_step", controller="hofa_policy_residual",
                       route=self.reference.route["name"], finish_reason=reason,
                       duration_s=self.elapsed, reward=self.reward_total, steps=self.steps,
                       completion_ratio=float(self.reference.arc/p["total_length"]),
                       final_distance_to_goal_m=float(np.hypot(self.state["x"]-goal[0],self.state["y"]-goal[1])),
                       cross_track_error_m=stats("cross_track_m"), yaw_error_deg=stats("yaw_error_deg"),
                       turn_yaw_error_deg=stats("yaw_error_deg",True),
                       turn_yaw_rate_error_radps=stats("yaw_rate_error",True),
                       saturation_ratio=float(np.mean([r["saturated"] for r in rows])) if rows else 0.0)
        metrics["comparison_reward"] = float(sum(row["comparison_reward"] for row in rows))
        metrics["comparison_reward_components"] = {
            key: float(sum(row["comparison_reward_"+key] for row in rows))
            for key in ("progress", "cross", "yaw", "yaw_rate", "turn_yaw_rate",
                        "terminal_yaw", "potential_terminal_yaw",
                        "potential_turn_yaw_rate", "potential", "lag", "effort",
                        "smooth", "desired_smooth", "time", "terminal")}
        for name, key in (("cross_track_p95_abs_m", "cross_track_m"),
                          ("yaw_p95_abs_deg", "yaw_error_deg")):
            metrics[name] = float(np.percentile([abs(row[key]) for row in rows], 95)) if rows else None
        turns = [row for row in rows if abs(row["reference_yaw_rate"]) >= 0.05]
        metrics["turn_yaw_p95_abs_deg"] = float(np.percentile(
            [abs(row["yaw_error_deg"]) for row in turns], 95)) if turns else None
        metrics["final_yaw_error_deg"] = float(np.degrees(np.arctan2(
            np.sin(self.state["yaw"]-p["yaw"][-1]), np.cos(self.state["yaw"]-p["yaw"][-1]))))
        if rows:
            applied = np.asarray([[row["residual_"+axis] for axis in ("fx", "fy", "nz")] for row in rows])
            changes = np.diff(np.vstack((np.zeros(3), applied)), axis=0)
            metrics["residual_rms"] = np.sqrt(np.mean(applied**2, axis=0)).tolist()
            metrics["residual_rate_rms_per_s"] = np.sqrt(np.mean((changes/self.dt)**2, axis=0)).tolist()
            metrics["residual_limit_hit_fraction"] = np.mean(np.abs(applied) >= 0.99*self.limits, axis=0).tolist()
            metrics["desired_smooth_reward_sum"] = float(sum(r["reward_desired_smooth"] for r in rows))
            # Report command jitter separately in turns and non-turn sections.
            # Empty sections remain missing, rather than becoming zero jitter.
            for suffix, section in (("", rows), ("_turn", turns),
                                    ("_nonturn", [r for r in rows if abs(r["reference_yaw_rate"]) < 0.05])):
                metrics["action_diagnostic_samples"+suffix] = len(section)
                for field, name in (("desired_change", "desired_residual_step_rms"),
                                    ("residual_change", "applied_residual_step_rms")):
                    values = np.asarray([[r[field+"_"+a] for a in ("fx","fy","nz")] for r in section])
                    metrics[name+suffix] = np.sqrt(np.mean(values**2, axis=0)).tolist() if section else None
                metrics["residual_slew_limited_fraction"+suffix] = [float(np.mean(
                    [r["slew_limited_"+a] for r in section])) for a in ("fx","fy","nz")] if section else None
        summary = self.output/"episodes.jsonl"
        with summary.open("a") as stream:
            stream.write(json.dumps(dict(episode=self.episode, **metrics))+"\n")
        if self.record_steps:
            dest = self.output/("episode_%06d" % self.episode)
            dest.mkdir()
            (dest/"metrics.json").write_text(json.dumps(metrics,indent=2)+"\n")
            (dest/"route.json").write_text(json.dumps(self.reference.route,indent=2)+"\n")
            write_csv(dest/"residual_steps.csv",rows)
            write_csv(dest/"odometry.csv",self.odom_rows)
            write_csv(dest/"generalized_force.csv",self.force_rows)
            write_csv(dest/"pwm.csv",self.pwm_rows)
            write_csv(dest/"reference_window.csv",self.window_rows)
            write_csv(dest/"reference_profile.csv",[
                dict(s_m=float(p["s"][i]), x=float(p["x"][i]), y=float(p["y"][i]),
                     yaw=float(p["yaw"][i]), curvature=float(p["curvature"][i]),
                     speed=float(p["speed"][i]), time_s=float(p["time"][i]),
                     yaw_rate=float(p["speed"][i]*p["curvature"][i])) for i in range(len(p["s"]))])
        return metrics

    def close(self):
        if self.active:
            self._finish("closed_interrupted")
        if self.backend is not None:
            self.backend.close()
