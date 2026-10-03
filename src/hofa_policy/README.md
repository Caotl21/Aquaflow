# hofa_policy

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

当前代码是 P0 工程骨架。正式闭环前仍需完成 Shadow、独立路线测试、MPC 回退和 MCU 一致性核对。

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
