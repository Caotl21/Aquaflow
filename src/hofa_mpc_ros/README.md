# BricsBot 跟踪实验脚本

运行一次 BricsBot + Stonefish + HOFA-MPC 实验，并自动保存结果：

```bash
cd ~/catkin_ws
catkin_make
source devel/setup.bash
rosrun hofa_mpc_ros run_mpc_experiment.py --scene empty_pool
```

默认启动 `hofa_mpc`。使用双环 PID 后端时：

```bash
rosrun hofa_mpc_ros run_mpc_experiment.py \
  --controller pid --scene empty_pool \
  --goal-x 4.0 --goal-y 0.0 --timeout 60
```

无窗口运行 Stonefish：

```bash
rosrun hofa_mpc_ros run_mpc_experiment.py --scene empty_pool --headless
```

默认目标为 `world_ned` 中的 `(x=4.0, y=0.0)`，结果写入：

```text
results/mpc/run_YYYYMMDD_HHMMSS/
```

可选参数：

```bash
rosrun hofa_mpc_ros run_mpc_experiment.py \
  --controller pid --scene random_pillars --goal-x 5.0 --goal-y -2.0 \
  --timeout 120 --output-root results/mpc
```

如果已经手动启动仿真和控制器：

```bash
rosrun hofa_mpc_ros run_mpc_experiment.py --no-launch
```

脚本会保存 `odometry.csv`、`controller_status.csv`、`generalized_force.csv`、`pwm.csv`、`mpc_reference_window.csv`、`evaluation_reference.csv`、`global_reference_path.csv`、`metrics.json`、`summary.txt` 和 `runtime_config.json`，并在安装了 matplotlib 时生成 `tracking_xy.png`、`tracking_errors.png`、`speed_tracking.png` 和 `control_wrench.png`。默认还会保存 `topics.bag`。

评测参考由 `hofa_mpc_ros.arc_profile.build_arc_profile` 用全局路径和 `/reference_processor` 的实际限幅参数**离线重算**，得到一份绝对计划表（`evaluation_reference = global_path_speed_profile`）。这与控制器所用的是同一个剖面函数，但不依赖控制器的输出。

不要用 `mpc_reference_window.csv` 的 `index=0` 作为评测参考：该点由 `s_start = s_proj` 生成，即车自身在路径上的投影，因此 `planned_duration_s` 会恒等于本次运行时长、沿程进度误差恒为 0，时间维度上测不出任何东西。该 CSV 保留作诊断用途。

指标分两个互不混淆的维度：

- **几何（按位置索引）**：`cross_track_error_m`、`geometric_path_error_m`，以及 `yaw_error_deg` / `speed_error_mps` —— 后两者对比的是剖面在**车当前弧长位置**处的切向和速度，回答"在这个位置上姿态和速度对不对"。
- **时序（按计划索引）**：`progress_error_m`（当前弧长 − 计划弧长）、`schedule_lag_s`（正=落后，负=超前）、`time_lag_at_finish_s`（在车实际到达的弧长处的时间偏差，未跑完也有定义）、`completion_ratio`。

另有 `speed_limit_violation_ratio`、`mean_speed_mps`、`max_speed_mps` 用于直接暴露超速，`final_distance_to_goal_m` 作为 `finish_reason` 之外的连续量，以及 `turn_yaw_error_deg` / `turn_yaw_rate_error_radps` 用于单独评估转弯阶段表现。

已有实验可用新口径离线重算，不需要仿真器或 ROS master：

```bash
rosrun hofa_mpc_ros run_mpc_experiment.py --rescore results/pid/run_20260910_122124
```

它从目录内的 CSV 重建数据、重新判定 `finish_reason`，并覆写 `metrics.json`、`summary.txt` 和图。在线与离线走的是同一个 `compute_metrics`，口径保证一致。

`runtime_config.json` 在启动等待结束后生成，记录本次运行时从 ROS 参数服务器读取的实际参数、关键源码和 YAML 配置的 SHA256、Git commit、未提交改动、主机和 Python 信息。它用于确认实验是否真正加载了当前代码和参数，而不是依赖工作区中的旧 `log.txt`。

脚本将 Stonefish odometry 的 body-frame 速度转换为 world NED 速度后再计算速度指标。

`reference_processor` 的 `progress_mode` 控制参考锚点：`schedule`（默认）按速度剖面积分推进绝对计划进度 `s_desired(t)`，控制器因此能看到真实的纵向误差；`projection` 是旧行为，每周期把参考重锚定到车自身投影。`max_schedule_lead_m` 限制计划最多领先车多少米，防止车卡住时参考跑飞。在线可用 `/controller/schedule_lag_s` 观察追赶过程。A/B 对比：

```bash
roslaunch hofa_mpc_ros simulation.launch controller:=pid progress_mode:=projection
```
