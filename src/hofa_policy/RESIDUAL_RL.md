# Frozen DAgger + residual TD3/SAC

在 Hydrocore 容器内执行以下命令。训练使用真实 Stonefish 1.6 引擎、现有
`aquaflow_empty_pool_control.scn` 和六推进器物理模型；使用底层固定步进接口，
不会在神经网络优化期间推进仿真。训练无需 ROS master，ROS 用于独立闭环复现。
这条原生同步链与 ROS 链的调度/状态采样方式不同，不能未经复现便把两者指标混用。

## 安装和构建

```bash
cd /home/bricsbot/catkin_ws
bash scripts/setup_hofa_residual_rl.sh
bash scripts/run_hofa_residual_rl.sh smoke \
  --output results/hofa_policy/residual_rl/smoke
```

安装在工作区 `.venv-hofa-rl`，继承现有 oceangym 的 Torch，安装
Stable-Baselines3 2.7.0 / Gymnasium 1.2.0；不修改原 oceangym 的包。
适配器编译到 `artifacts/hofa_rl_native_hydrocore/`。可通过
`HOFA_RL_BASE_PYTHON` 和 `HOFA_STONEFISH_PREFIX` 指定本机 Python 和引擎安装位置。
容器与宿主的 C++ 二进制不可混用，需要分别编译。

smoke 检查 Gym API、真实物理固定时间步、重置可复现性和 DAgger 权重不变。
它不是策略性能验收，也不代表 ROS 原始 DAgger 指标已复现。

## 分别训练

```bash
bash scripts/run_hofa_residual_rl.sh train \
  --algo td3 --seed 0 --total-timesteps 100000 --device auto \
  --output results/hofa_policy/residual_rl/td3_seed0

bash scripts/run_hofa_residual_rl.sh train \
  --algo sac --seed 0 --total-timesteps 100000 --device auto \
  --output results/hofa_policy/residual_rl/sac_seed0
```

`config/residual_rl.json` 是两种算法共用的环境、残差范围、奖励和超参数入口。
默认冻结 `results/hofa_policy/train_dagger_v1/best.pt`。动作是机体系
`[delta_Fx, delta_Fy, delta_Nz]`，初始硬上限为 `[1.0 N, 0.4 N, 0.08 N m]`，
并限制每步变化量。20 步参考窗口、归一化和历史动作契约与 ROS 部署共用。
深度继续使用 PID，RL 只修正水平控制。冻结基线不能保证组合闭环稳定。

TD3 的初始确定性动作是零；SAC 的初始均值是零、方差较小。
warmup 和探索仍会产生受限残差。训练场景在三个路径家族内作长度和镜像变化。
验证用固定的另一组扰动种子 10000 起；最终测试用固定原始路径和 21000 起的扰动种子。
这是同路径家族的泛化检查，不代表未知路径家族或真实机器人的泛化。

输出包括 `final.zip`、周期 checkpoint、TensorBoard、训练配置、冻结基线与
物理场景/源码哈希、逐 episode 指标。每 5000 步用独立 Stonefish 进程验证并保存 checkpoint，
验证回报最高的模型另存 `best.zip`。验证尚未触发时没有 `best.zip`。
可用 `--record-train-steps` 保存完整训练轨迹，用 `--save-replay-buffer` 保存经验池。
初版不支持从 checkpoint 恢复训练；已有输出目录会拒绝覆盖。

三种种子、两种算法统一预算：

```bash
bash scripts/train_hofa_residual_suite.sh results/hofa_policy/residual_rl/suite_100k 100000
```

不传参数的一键默认命令会使用 100,000 步、seed `0,1,2`、TD3/SAC 各三次，
依次训练共六个 run：

```bash
bash scripts/train_hofa_residual_suite.sh
```

默认 `--device auto` 会在训练器进程中检查 CUDA，并在 `training.json` 与
`completed.json` 记录实际设备和 GPU 型号；CUDA 不可用时自动用 CPU。
显式 `cuda` 但 Docker 没有 GPU 访问权限会在开跑前报错。DAgger 推理和
Stonefish 物理仍在 CPU 上，TD3/SAC actor、critic 更新使用所选设备。每个 run
默认每 5,000 步独立验证并存 checkpoint；完整日志在 suite 根目录。

参数顺序为 `输出目录 步数 seed列表 device 验证间隔 checkpoint间隔`，例如：

```bash
bash scripts/train_hofa_residual_suite.sh \
  results/hofa_policy/residual_rl/suite_100k 100000 0,1,2 auto 5000 5000
```

已完成的 seed 会自动跳过；不完整目录会报错，需另选一个输出目录继续，
训练器当前不从中断 checkpoint 恢复 replay buffer。

### 六路正式并行训练

2026-10-21 的 CUDA 短测中，2/4/6 路总吞吐分别为 242.6/397.9/514.0 steps/s，
各档全部成功，因此这台机器选 6 路。原始记录位于
`results/hofa_policy/residual_rl/parallel_benchmark_20261021_080856/`。

正式训练可使用独立的并行入口，默认 TD3/SAC 各三个 seed、每组 100,000 步、
6 路并发、CUDA、每进程 1 个 CPU 数值库线程，每 5,000 步验证和保存 checkpoint：

```bash
python3 scripts/train_hofa_residual_parallel.py
```

默认创建带时间戳的新目录。也可沿用 suite shell 入口，第七个参数指定并发数：

```bash
bash scripts/train_hofa_residual_suite.sh \
  results/hofa_policy/residual_rl/suite_100k_parallel6 \
  100000 0,1,2 cuda 5000 5000 6
```

并发数小于总任务数时自动排队，仍执行全部 seed；每组独立持有模型、经验池和
训练/验证 Stonefish 进程。`suite.json` 与 `summary.csv` 记录命令、PID、状态和
完成步数，独立日志为 `td3_seed_0.log` 等，`gpu.csv` 每 5 秒记录 GPU 指标。
共享配置在启动前冻结到 suite 根目录 `config.json`，每组仍保存自己的配置与来源哈希。
并行入口拒绝任何已有输出目录；任一组失败会停止其他组及尚未启动的任务，Ctrl-C
会清理本次启动的训练/物理进程并保留已有产物。它不支持中断续训。

可先预览任务而不加载 Torch 或启动仿真：

```bash
python3 scripts/train_hofa_residual_parallel.py --dry-run
```

按短测总吞吐线性外推，六组各 10 万步的纯吞吐量级约为 19.5 分钟；实际还受
验证、checkpoint、经验池填满后的内存负载和长期散热影响，不能把该外推当作完成时间保证。
正式并行训练保留验证，按验证回报选 `best.zip`，训练后仍需独立测试及 ROS 闭环复现。

## 并行吞吐测试

在一次正式的六组训练之前，可以用短训练比较 2、4、6 组独立任务同时运行时的
总吞吐。每个任务有自己的 Python、经验池、Stonefish 子进程和输出目录；它们不共享
经验数据或模型权重。默认每任务训练 20,000 步，关闭验证和 checkpoint，因此结果仅
用于选择并发数，不能用于比较策略效果。

```bash
python3 scripts/benchmark_hofa_residual_parallel.py
```

默认输出到一个新的、带时间戳的目录。可先用 10,000 步快速摸底，或只测 6 路：

```bash
python3 scripts/benchmark_hofa_residual_parallel.py \
  --steps 10000 --levels 2,4,6 --device cuda

python3 scripts/benchmark_hofa_residual_parallel.py \
  --steps 20000 --levels 6 --device cuda
```

输出根目录中的 `summary.csv` 给出每一档并发的总 wall-clock FPS，`tasks.csv` 给出
每个任务的耗时和状态，`concurrency_N/gpu.csv` 每秒记录 GPU 利用率、显存、功耗和
温度。应按 `aggregate_fps` 选总吞吐最高且所有任务成功的档位；短训练中产生的
`final.zip` 不应用于策略选型。Ctrl-C 会终止当前档位中所有训练和 Stonefish 进程，
已完成的档位及日志会保留。

wall-clock FPS 包括进程启动、默认前 2,000 步随机采样、训练、最终模型保存和退出。
每档使用相同比例的 TD3/SAC，但任务数不同：2 路测 seed 0，4 路测 seed 0/1，
6 路测 seed 0/1/2。建议每任务至少 10,000–20,000 步；小于 2,000 步不能测到
网络更新负载。短测试也不能代表完整 200,000 容量经验池的内存峰值或长时间热稳定性。
运行时应避免同时运行其他正式训练，否则总 FPS 和 GPU 采样会受到其他任务影响。
任意档位有失败就停止后续档位；已有输出目录始终拒绝覆盖。日志和命令文件放在
`concurrency_N/td3_seed_0.log`、`td3_seed_0.command.json` 等位置，模型输出放在
对应的 `td3_seed_0/` 子目录。

## 原生环境独立评测和画图

对一套完整训练结果，推荐使用独立验证入口一次性评估 DAgger、六个训练 seed
和相同的固定/随机测试案例。默认每条路线使用 1 个固定案例和 3 个带固定种子的
随机案例，共 12 个 episode/方法；入口会拒绝缺失 checkpoint 或已有输出目录：

```bash
bash scripts/evaluate_hofa_residual_suite.sh \
  --suite results/hofa_policy/residual_rl/suite_parallel_20261021_094553 \
  --random-cases 3 \
  --output results/hofa_policy/residual_rl/independent_eval_suite094553
```

它评估 `dagger`、`td3_seed_0..2`、`sac_seed_0..2` 的 `best.zip`，输出
`episodes.csv/json`（逐回合指标）、`summary.csv/json`（方法级汇总）、
`paired_deltas.csv/json`（每个相同测试案例相对 DAgger 的差值）、每个方法的原始
`evaluation.json`、逐回合图和按案例的对比图。`--checkpoint final.zip` 可改用最终
迭代模型；正式比较默认使用验证回报最高的 `best.zip`，并在 `manifest.json` 中记录
案例种子、suite、配置和评估器。可先检查路径而不启动仿真：

```bash
bash scripts/evaluate_hofa_residual_suite.sh \
  --suite results/hofa_policy/residual_rl/suite_parallel_20261021_094553 \
  --random-cases 3 --dry-run
```

```bash
bash scripts/run_hofa_residual_rl.sh compare \
  --td3-model results/hofa_policy/residual_rl/td3_seed0/best.zip \
  --sac-model results/hofa_policy/residual_rl/sac_seed0/best.zip \
  --random-cases 3 \
  --output results/hofa_policy/residual_rl/comparison_seed0
```

同时评测零残差 DAgger、TD3、SAC；使用相同测试路径、确定性动作和种子。
不提供模型参数时只跑零残差基线。输出逐回合 CSV、参考窗口、基础/残差/组合
动作、PWM、各奖励分量、误差指标、路径与误差 PNG/PDF 图和汇总 JSON/CSV。
模型算法、配置、基线哈希不一致会拒绝评测。

## 在原 ROS 链中复现

```bash
source scripts/setup_stonefish_ros.sh
rosrun hofa_mpc_ros run_tracking_experiment.py \
  --controller hofa_policy --route right_angle --headless \
  --checkpoint /home/bricsbot/catkin_ws/results/hofa_policy/train_dagger_v1/best.pt \
  --residual-algo td3 \
  --residual-model /home/bricsbot/catkin_ws/results/hofa_policy/residual_rl/td3_seed0/best.zip
```

SAC 只需替换 `--residual-algo sac` 和对应模型路径。脚本自动选工作区 RL
虚拟环境运行控制节点；有其他路径时可显式传 `--policy-python`。
RViz 手动模式同样可在 `simulation.launch` 指定 `policy_residual_model`、
`policy_residual_algo` 和 `policy_python`（工作区 `.venv-hofa-rl/bin/python`）。
对比中需要相同参考限速；训练默认 0.2 m/s，手动 launch 也应设置
`reference_max_speed:=0.2`。残差配置本身不修改 ROS 参考处理器参数。

最终以多训练种子的成功率、急弯航向/角速度误差、横向误差、完成时间、
安全停止和推进器饱和率判断改善；短训练的成功运行不代表效果超过 DAgger。

## MPC、DAgger 与 C-seed0 的 ROS 三路配对复现

reserved-final 通过后，可用原 ROS/Stonefish 控制链在完全相同的路线和仿真参数下
复现 MPC 教师、零残差 DAgger 与 `C_uniform_tolerance_sac_seed_0`。脚本从
reserved-final manifest 核验 DAgger checkpoint 和候选 residual checkpoint 的
SHA256，只评测这个已选模型；默认在三类路线各生成 3 个新 seed，共 9 个配对案例、
27 次新仿真。每个案例使用全新的 simulator，三种方法按重复编号循环轮换先后顺序。

```bash
cd ~/catkin_ws
catkin_make --pkg hofa_mpc_ros
source scripts/setup_stonefish_ros.sh
bash scripts/compare_hofa_residual_ros.sh --dry-run
bash scripts/compare_hofa_residual_ros.sh
```

结果默认写入 `results/hofa_policy/residual_rl/ros_paired_threeway_local_projection/`，包括
逐回合指标、三方法汇总、路线汇总、运行日志和三路对比图。`paired_deltas.csv/json`
表示 SAC 残差相对 DAgger 的逐案例差值，`teacher_paired_deltas.csv/json` 表示
SAC 残差相对 MPC 教师的逐案例差值，`verdict.json` 中的 `teacher_comparison`
保存 MPC 的总体对照。总体通过判定仍只检查候选相对 DAgger 的安全门槛；MPC 对照
单独报告，不把教师和蒸馏策略混成一个 baseline。终点 yaw 继续记录为诊断项，
不影响通过判定。中断后使用同一命令并加 `--resume` 继续；已完成的运行会复用。
如果需要重新计算已经完成的报告，不必重跑仿真：

```bash
bash scripts/regrade_hofa_residual_ros.sh
```

结果只代表这三类路线的仿真复现，
不代表新路线泛化或实艇表现。

成对实验会记录 `reference_processor.py` 和
`config/reference_processor.yaml` 的 SHA256，并使用当前弧长附近的局部投影，
避免 Hairpin 自交路径在去程/回程分支之间跳转。

## Residual RL 2×2 筛查实验

上一版 residual RL 的独立评测没有整体超过零残差 DAgger。下一轮先不增加网络
规模、不延长到百万步，也不把绝对误差 reward 替换成纯 improvement reward。
它固定 SAC、`[128,128]` 网络、物理场景、残差上限、正则项、熵机制、seed 和
训练预算，仅比较以下两个因素：

| Group | warmup 前 2,000 步 | tracking reward |
|---|---|---|
| A | 原 SB3 全范围均匀动作 | 旧版绝对误差 reward |
| B | 零中心截断高斯，`std=0.05`、clip `±0.15` | 旧版绝对误差 reward |
| C | 原 SB3 全范围均匀动作 | 工程容差归一化的绝对误差 reward |
| D | 零中心截断高斯，`std=0.05`、clip `±0.15` | 工程容差归一化的绝对误差 reward |

容差组将横向误差尺度设为 `1 cm`、航向误差尺度设为 `5°`；角速度和 lag 的
尺度保持 `0.5 rad/s`、`1 m`。这些是待验证的起点，并非已经证明最优的阈值。
为使“减小 residual 上限”的后续消融可解释，所有组的 residual magnitude/slew
正则分母固定在 `[1.0, 0.4, 0.08]` 和 `[0.3, 0.15, 0.025]`，不随本轮变化。
完整计划位于 `src/hofa_policy/config/residual_ablation.json`。

## 终点航向与转弯角速度 reward 小规模消融

三路 ROS 复现表明，SAC 残差已经改善 DAgger 的位置和转弯航向误差，但终点航向和
转弯航向角速度仍落后 MPC。下一轮固定 SAC、`[128,128]` 网络、uniform residual
warmup、tolerance tracking scales、残差上限、slew 限制、物理正则、seed 和训练
预算，只消融两个 gated reward 项：

| Group | 终点航向项 | 转弯航向角速度项 |
|---|---:|---:|
| A_current_reward | 关闭 | 关闭 |
| B_terminal_yaw | 开启 | 关闭 |
| C_turn_yaw_rate | 关闭 | 开启 |
| D_terminal_yaw_turn_yaw_rate | 开启 | 开启 |

终点项只有在同时满足距离终点不超过 `0.5 m`、剩余轨迹弧长不超过 `0.5 m` 时生效，
避免 Hairpin 交叉段提前触发；转弯角速度项只在参考航向角速度绝对值不小于
`0.05 rad/s` 时生效。起始权重为终点航向 `0.15`、转弯角速度 `0.10`，是保守的
待验证起点。每组使用 seed `0,1,2`、50,000 步，每 5,000 步验证并保存 checkpoint，
共 12 个 SAC run、6 路并行。验证中的模型选择继续使用共同的当前 tolerance reward，
避免直接按不同训练 reward 的数值比较组间结果。

先预览任务，再启动训练：

```bash
cd ~/catkin_ws
bash scripts/train_hofa_residual_reward_ablation.sh --dry-run
bash scripts/train_hofa_residual_reward_ablation.sh
```

也可以显式指定输出目录，避免时间戳目录不便记录：

```bash
bash scripts/train_hofa_residual_reward_ablation.sh \
  --output results/hofa_policy/residual_rl/reward_ablation_50k \
  --device cuda --workers 6
```

训练完成后，输出根目录的 `suite.json`、`summary.csv`、`gpu.csv` 和
`reward_ablation.json` 记录任务状态、硬件、冻结计划和来源；每组目录中有独立的
`config.json`，每个 seed 目录中有 `best.zip`、`final.zip`、`validation.jsonl`、
`selection.json` 和 checkpoint。该实验仍需用独立 screening split 评估后，才能决定
是否进入 reserved-final 和 ROS 三路复现。

## 平滑势函数差分 reward 消融

上一轮硬门控 reward 的组间差异有限，因此新版本将两个附加项改为势函数差分：

```text
r_potential = gamma * Phi(state_next) - Phi(state)
```

终点航向势函数随剩余弧长平滑增加，转弯航向角速度势函数随参考航向角速度从
`0.02` 到 `0.20 rad/s` 平滑增加。这样每一步都能提供方向一致的 shaping 信号，
不会在某个距离阈值处突然打开惩罚。四组为：

| Group | 终点航向势 | 转弯角速度势 |
|---|---:|---:|
| A_current_reward | 关闭 | 关闭 |
| B_potential_terminal_yaw | 开启 | 关闭 |
| C_potential_turn_yaw_rate | 关闭 | 开启 |
| D_potential_both | 开启 | 开启 |

默认势函数权重为终点航向 `0.15`、转弯角速度 `0.10`，其余网络、warmup、残差
限制、正则项、seed、训练预算和验证选择规则保持一致。验证选择仍使用禁用势函数的
共同 comparison reward，避免按不同训练目标的 return 直接比较模型。

配置和入口为：

```bash
cd ~/catkin_ws
bash scripts/train_hofa_residual_potential_ablation.sh --dry-run
bash scripts/train_hofa_residual_potential_ablation.sh \
  --output results/hofa_policy/residual_rl/potential_ablation_50k \
  --device cuda --workers 6
```

这会启动 12 个 SAC run（4 组 × 3 seed），每个 50,000 步，每 5,000 步验证并保存。
输出根目录记录 `suite.json`、`potential_ablation.json`、`summary.csv` 和 GPU 日志；
每个 seed 目录保存 `best.zip`、`final.zip`、`validation.jsonl`、`selection.json`。
训练完成后应先在 `31000` screening split 上评估，再决定是否进入 reserved-final 和
ROS 三路复现。

100 万步 A/B 长实验使用专用筛查入口（旧的 2×2 evaluator 不识别该 suite schema）：

```bash
bash scripts/evaluate_hofa_residual_potential_screening.sh --dry-run \
  --experiment results/hofa_policy/residual_rl/potential_long_1m_20261025 \
  --output results/hofa_policy/residual_rl/potential_long_1m_20261025_screening

bash scripts/evaluate_hofa_residual_potential_screening.sh \
  --experiment results/hofa_policy/residual_rl/potential_long_1m_20261025 \
  --output results/hofa_policy/residual_rl/potential_long_1m_20261025_screening
```

该筛查读取 DAgger 和 6 个已有 `best.zip`，共 63 个 episode，只使用 `31000` 系列
案例；不会重新训练，也不会读取 `41000` reserved-final 案例。

## 1M 候选的 PID/MPC/DAgger/RL 四路 ROS 对照

`compare_hofa_residual_ros.sh` 的三路版本是 MPC、DAgger 和 RL residual。若要把
100 万步筛选出的 `A_current_reward/sac_seed_1/best.zip` 与 PID、MPC、DAgger 一起
比较，使用专用四路入口：

```bash
bash scripts/compare_hofa_all_controllers_ros.sh --dry-run \
  --experiment results/hofa_policy/residual_rl/potential_long_1m_20261025 \
  --screening results/hofa_policy/residual_rl/potential_long_1m_20261025_screening

bash scripts/compare_hofa_all_controllers_ros.sh \
  --experiment results/hofa_policy/residual_rl/potential_long_1m_20261025 \
  --screening results/hofa_policy/residual_rl/potential_long_1m_20261025_screening \
  --output-root results/hofa_policy/residual_rl/ros_paired_fourway_1m_A_seed1
```

默认使用 9 个新的 `61000` 系列随机路线案例，共 36 次仿真；四种方法共享路线 JSON、
场景、参考限幅和局部弧长投影，方法顺序按案例轮换。该报告只做描述性比较，不在
这组 ROS 案例上重新选择模型；中断后可加 `--resume` 续跑。

## 上一版 2×2 计划的执行与筛查

每组使用 seed `0,1,2`，各训练 50,000 步；合计 12 个 SAC run。每 5,000 步
进行一次固定的 9-case 验证，并保存 checkpoint。验证同时运行零 residual DAgger，
并按共同的旧 reward 和物理指标选择 `best.zip`：候选必须保持成功率、航向/转弯
误差不超过 DAgger 的 5% 相对退化、终点距离不多于 `5 mm`、完成时间不多于
`0.5 s`，且平均横向 RMSE 至少改善 `0.1 mm`；否则该 run 的选择结果为
`zero_residual`。训练 reward 仅用于各自算法学习，不能跨组排序。

在 Hydrocore 中先预览，再启动训练：

```bash
cd /home/bricsbot/catkin_ws

bash scripts/train_hofa_residual_ablation.sh --dry-run

bash scripts/train_hofa_residual_ablation.sh \
  --output results/hofa_policy/residual_rl/ablation_reward_warmup_v1
```

默认 6 路并发，因此 12 个 run 分两批执行。每个 group/run 分别保存冻结配置、
`selection.json`、验证曲线、checkpoint、GPU 采样、日志和完整来源哈希；任一 run
失败会停止本批作业并保留状态。

训练结束后，在与验证 seed 不重叠的 12 个 development screening cases 上评测
DAgger 和全部 12 个候选：

```bash
bash scripts/evaluate_hofa_residual_ablation.sh \
  --experiment results/hofa_policy/residual_rl/ablation_reward_warmup_v1 \
  --output results/hofa_policy/residual_rl/ablation_reward_warmup_v1_screening
```

该命令使用所有组共同的旧 reward 输出 `episodes`、`summary`、`group_summary`、
`paired_deltas`、`factor_effects` 和 `screening_selection.json`；后者只在候选先通过
验证 gate 后才允许其击败零 residual。计划中还预留了 15 个未使用的 final cases，
在决定一个具体候选前不运行，以避免用最终测试集选模型。
