# HOFA-MPC 实验脚本

运行一次 BricsBot + Stonefish + HOFA-MPC 实验，并自动保存结果：

```bash
cd ~/catkin_ws
catkin_make
source devel/setup.bash
rosrun hofa_mpc_ros run_mpc_experiment.py --scene empty_pool
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
  --scene random_pillars --goal-x 5.0 --goal-y -2.0 \
  --timeout 120 --output-root results/mpc
```

如果已经手动启动仿真和控制器：

```bash
rosrun hofa_mpc_ros run_mpc_experiment.py --no-launch
```

脚本会保存 `odometry.csv`、`controller_status.csv`、`generalized_force.csv`、`pwm.csv`、`mpc_reference_window.csv`、`global_reference_path.csv`、`metrics.json`、`summary.txt`，并在安装了 matplotlib 时生成 `tracking_xy.png`、`tracking_errors.png`、`speed_tracking.png` 和 `control_wrench.png`。默认还会保存 `topics.bag`。

指标包括横向误差、几何路径误差、yaw 误差、速度误差、沿程进度误差、求解成功率、回调耗时和 PWM 饱和比例。脚本将 Stonefish odometry 的 body-frame 速度转换为 world NED 速度后再计算速度指标。
