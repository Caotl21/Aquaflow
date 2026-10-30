# hofa_policy

冻结 DAgger 后使用 TD3/SAC 训练小残差控制器，见 [RESIDUAL_RL.md](RESIDUAL_RL.md)。

水平面 HOFA-MPC 策略蒸馏包。它与规划层 `aquaflow_bc` 相互独立。

第一版从已有 MPC 实验记录构造训练数据，学习：

```text
当前机体系速度 + 20 步局部参考 + 上一广义力
                         → 当前可达水平广义力 [Fx,Fy,Nz]
```

网络内部输出 4 个有界变量，通过固定推进器范围和分配矩阵合成广义力。ROS/MCU 外部接口保持 `/controller/generalized_force` 不变。

## 导出一个 teacher run

```bash
cd /home/bricsbot/catkin_ws
export PYTHONPATH="$PWD/src/hofa_policy:$PYTHONPATH"
/home/bricsbot/miniconda3/envs/oceangym/bin/python -m hofa_policy.export_run \
  --run results/mpc/run_20260926_161539 \
  --output results/hofa_policy/datasets/run_20260926_161539
```

## 训练

先用单个 run 做 128 样本工程过拟合。该模式故意让训练和验证使用同一批样本，只检查数据与学习链路，不代表泛化能力：

```bash
/home/bricsbot/miniconda3/envs/oceangym/bin/python -m hofa_policy.train \
  --train results/hofa_policy/datasets/run_20260926_161539 \
  --output results/hofa_policy/overfit_128_run_20260926_161539 \
  --mode overfit --overfit-samples 128 \
  --epochs 300 --batch-size 128 --device cuda
```

正式训练和验证必须使用不同的完整 run：

```bash
/home/bricsbot/miniconda3/envs/oceangym/bin/python -m hofa_policy.train \
  --train results/hofa_policy/datasets/train_run_a \
          results/hofa_policy/datasets/train_run_b \
  --val results/hofa_policy/datasets/val_run_a \
  --output results/hofa_policy/train_01 \
  --epochs 100 --device cuda
```

## RViz 点目标闭环

策略已接入 `hofa_mpc_ros/simulation.launch` 的控制后端选择：

```bash
source /home/bricsbot/catkin_ws/scripts/setup_stonefish_ros.sh
roslaunch hofa_mpc_ros simulation.launch controller:=hofa_policy \
  policy_checkpoint:=/home/bricsbot/catkin_ws/results/hofa_policy/train_v1_epoch500/best.pt \
  policy_device:=cpu
```

另一个已加载 ROS 环境的终端运行：

```bash
roslaunch aquaflow_stonefish visualization.launch vehicle_model:=bricsbot
```

RViz 的 `2D Nav Goal` 经 A*、平滑和参考处理器生成策略输入，策略发布
`/controller/generalized_force`，六推进器分配器输出 PWM。手动启动时同样
保持 `experiment_depth`（默认 1 m），无需设置实验启动同步 topic。
`simulation.launch` 已包含教师节点，不要另开 `teacher.launch`。

`controller:=pid` 和 `controller:=hofa_mpc` 可切换为另外两种后端。
已有 Shadow 和无障碍独立路线闭环记录；随机障碍路线、MPC 自动回退和
MCU 一致性仍需单独验证。

## BC 与 DAgger 同路径对比

在同一条急弯路径、同一场景和同一参考参数下顺序测试原始 BC 与 DAgger：

```bash
source /home/bricsbot/catkin_ws/scripts/setup_stonefish_ros.sh
rosrun hofa_policy run_policy_comparison.py \
  --route right_angle \
  --baseline-checkpoint /home/bricsbot/catkin_ws/results/hofa_policy/train_v1_epoch500/best.pt \
  --dagger-checkpoint /home/bricsbot/catkin_ws/results/hofa_policy/train_dagger_v1/best.pt
```

每个 checkpoint 都会启动一次全新的 Stonefish，结果分别保存在
`results/hofa_policy/policy_comparison/baseline_bc/` 和 `dagger/`；汇总结果
写入同一目录下的 `comparison_*.json` 与 `comparison_*.csv`，并自动生成
`plots/comparison_xy.*`、`plots/comparison_errors.*` 和
`plots/comparison_metrics.*`，每个 run 目录也保存单独的 `plots/`。可用
`--route hairpin` 或 `--route switchback` 测试其他固定路径，使用
`--headless` 关闭 Stonefish 图形窗口。

## Shadow 评估

HOFA-MPC 保持唯一控制输出，策略只发布 `/hofa_policy/shadow_*` 诊断话题：

```bash
python3 src/hofa_mpc_ros/scripts/run_mpc_experiment.py \
  --output-root results/hofa_policy/shadow_v1/test_00_line \
  --controller hofa_mpc --scene empty_pool_control --headless --no-bag \
  --policy-trajectory results/hofa_policy/trajectory_suite_v1/tracks/test_00_line.json \
  --shadow-checkpoint results/hofa_policy/train_v1_epoch500/best.pt \
  --shadow-device cpu
```

运行完成后生成 Shadow 报告：

```bash
PYTHONPATH=src/hofa_policy/src \
/home/bricsbot/miniconda3/envs/oceangym/bin/python \
  -m hofa_policy.shadow_evaluate --run <实际 run_目录>
```

## 生成无障碍 teacher 轨迹套件

```bash
PYTHONPATH=src/hofa_policy/src \
/home/bricsbot/miniconda3/envs/oceangym/bin/python \
  -m hofa_policy.generate_suite \
  --output results/hofa_policy/trajectory_suite_v1 \
  --train 20 --val 5 --test 5 --seed 27
```

套件包含直线、S 弯、双 S、横移、chicane 和长弯，并为每条轨迹保存独立 split、速度和超时配置。

## 在 Hydrocore 中批量运行 HOFA-MPC teacher

先做一条 smoke run：

```bash
cd /home/bricsbot/catkin_ws
source scripts/setup_stonefish_ros.sh
python3 -m hofa_policy.collect_teacher \
  --suite results/hofa_policy/trajectory_suite_v1/suite.json \
  --output results/hofa_policy/teacher_v1 \
  --split train --limit 1
```

`hofa_policy` 使用独立的 `/hofa_policy/teacher_global_path`，不会占用 Aquaflow 规划层的 `/aquaflow/teacher_global_path`。

Shadow 运行需要让同一个 Python 环境同时加载 ROS Noetic 和 PyTorch。Hydrocore 的 `oceangym` 环境应安装 [requirements-shadow.txt](requirements-shadow.txt) 中的 `rospkg`、`catkin_pkg` 和 `netifaces`；系统 ROS Python 3.8 没有 PyTorch，不能直接加载策略 checkpoint。

批量采集默认使用 `aquaflow_empty_pool_control.scn` 和 camera-free `bricsbot_control_reference.scn`，因此可以运行 Stonefish `parsed_simulator_nogpu`。Aquaflow 仍使用带 `camera_front` 的原始 BricsBot 场景和渲染后端。`--rendered` 只用于控制采集链的可视化调试。每次失败尝试保留独立 `runner_NN.log`，不会覆盖之前的失败证据。

通过后去掉 `--limit 1`，再依次采集 train、val 和 test。采集器每条轨迹独立启动 Stonefish、运行 HOFA-MPC、保存原始实验 CSV，并自动导出 `hofa_policy_v1` 数据集。
