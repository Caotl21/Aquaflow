# hofa_policy 算法设计：BC → DAgger → RL 残差微调

本文整理 `hofa_policy` 当前代码、配置和实验计划中的控制策略设计。目标是回答五个问题：我们要替代什么、网络和数据如何组织、loss 如何定义、遇到的问题如何修复，以及在推进器输入受限和饱和时如何保持安全。

文中使用三个状态标记：

- **已实现**：代码路径已经存在，并且有明确的配置或接口。
- **已记录**：仓库文档或结果元数据记录过该事实，但本次整理没有重新跑完整实验。
- **待验收**：设计或脚本已经准备好，但还不能把它当成已经证明的闭环性能结论。

## 1. 总体目标和边界

`hofa_policy` 是水平面 HOFA-MPC 的策略蒸馏与微调包，和规划层 `aquaflow_bc` 独立。整体目标是：

```text
当前机体系状态 + 20 步局部参考 + 上一周期水平广义力
                         │
                         ▼
                  hofa_policy 网络
                         │
              4 路有界隐式推进器变量 q
                         │
        固定推进器范围与分配矩阵 B_h
                         │
           水平广义力 [Fx, Fy, Nz]
                         │
          /controller/generalized_force
                         │
              六推进器分配器与 PWM
```

设计边界如下：

1. 第一版只控制水平 3-DOF：`Fx`、`Fy`、`Nz`。深度方向仍由现有深度控制器处理。
2. 输入和输出遵守 NED/FRD 约定，ROS/MCU 外部接口仍是 `WrenchStamped` 的广义力。
3. 推进器几何、分配矩阵、限幅、坐标变换和消息格式由解析代码固定，不交给网络拟合。
4. MPC 在 teacher 数据生成、Shadow 对照和安全回退中继续保留。
5. RL 的当前形式是**冻结 DAgger 主策略上的小残差控制器**，不是从零学习一个绝对控制器。

## 2. 统一输入、输出和物理契约

契约定义在 [`contract.py`](src/hofa_policy/src/hofa_policy/contract.py) 和 [`features.py`](src/hofa_policy/src/hofa_policy/features.py)。控制周期为 10 Hz，参考间隔为 0.1 s，窗口长度为 20。

### 2.1 BC/DAgger 输入

每个样本对应一个控制周期：

| 输入 | 维度 | 内容 |
|---|---:|---|
| `state` | 3 | 当前机体系速度 `[u, v, r]` |
| `reference` | `20 × 10` | 每个参考点 `[x_body, y_body, sin(Δyaw), cos(Δyaw), vx_body, vy_body, dpsi, ax_body, ay_body, ddpsi]` |
| `previous_wrench` | 3 | 上一周期已发布的 `[Fx, Fy, Nz]` |

世界系参考在进入网络前转换到当前机体系；航向误差使用 `sin/cos`，避免角度在 `±π` 处跳变。特征构造是因果的：只使用动作时刻之前或同时可用的状态和参考。

### 2.2 BC/DAgger 标签与输出

teacher 标签是当前 HOFA-MPC 最终发布的水平广义力：

```text
target_wrench = [Fx, Fy, Nz]
```

标签已经包含原控制链中的 Layer 1、Layer 2、HOFA inverse、广义力缩放、安全限幅和坐标转换。策略不需要重新学习这些解析步骤。

网络最后输出 4 个内部变量 `q ∈ [-1, 1]^4`。固定执行器层按

```text
f = f_mid + f_half ⊙ q
τ = B_h f
```

得到水平广义力 `τ = [Fx, Fy, Nz]`。BricsBot 水平四推进器的单路范围是 `[-58.84, 67.67] N`；`B_h` 由推进器位置和方向计算，矩阵满秩覆盖平面 3-DOF。网络内部的 `f` 是物理可行性见证，外部仍只发送广义力。

### 2.3 归一化

模型内部固定使用以下尺度：

- 状态：`[0.5, 0.5, 1.2]`；
- 每个参考特征：`[2, 2, 1, 1, 0.5, 0.5, 1, 0.5, 0.5, 1]`；
- 广义力：由推进器几何计算 `wrench_scale`，近似覆盖可达集合。

这些尺度是模型契约的一部分，checkpoint 会保存契约和模型配置，加载时不允许输入/动作契约不一致。

## 3. 网络结构

实现位于 [`model.py`](src/hofa_policy/src/hofa_policy/model.py)，默认宽度来自 `ModelConfig`：

```text
20×10 reference
   └─逐点 MLP: 10 → 64 → 96
   └─1D Conv: 96 → 96 → 96
       ├─时间维 mean pooling
       └─时间维 max pooling

[state(3), previous_wrench(3)]
   └─context MLP: 6 → 64 → 64

[reference_mean(96), reference_max(96), context(64)]
   └─fusion MLP: 256 → 128 → 128 → 4
   └─tanh
   └─固定推进器映射和 B_h
```

使用 `mean + max` 是为了同时保留窗口的整体趋势和局部极值；上一周期广义力单独进入 context 分支，使网络能够看到控制历史并降低动作跳变。输出层不是任意的 `3` 维广义力回归，而是 `4` 维 `tanh` 后的推进器参数化。

## 4. 阶段一：Behavior Cloning（BC）

### 4.1 我们希望 BC 做什么

BC 的作用是从已有 HOFA-MPC 轨迹中学习一个低延迟的前向策略：在训练分布附近，给定状态、参考窗口和上一动作，复现 MPC 的水平广义力。它先解决“网络能否正确读入契约并产生物理可行输出”，再进入闭环和分布外问题。

BC 不承担以下任务：

- 不从原始图像直接规划路线；
- 不替代参考轨迹生成器；
- 不学习推进器几何和底层 PWM 曲线；
- 不通过随机拆帧的方式声称具备路线泛化能力。

### 4.2 teacher run 如何变成数据

[`export_run.py`](src/hofa_policy/src/hofa_policy/export_run.py) 从一个 MPC run 读取：

```text
odometry.csv
mpc_reference_window.csv
generalized_force.csv
controller_status.csv
```

对每个候选动作按以下因果条件筛选：

1. 找到不晚于动作时间戳的最近 odometry；年龄不超过 50 ms。
2. 找到不晚于动作时间戳的最近完整 20 点参考窗口；年龄不超过 150 ms。
3. controller 状态为 `ACTIVE`，并且 MPC solver 成功。
4. 参考窗口的索引必须连续。
5. 不满足条件的样本计入 `rejected`，不会静默混入训练集。

导出为 `samples.npz + metadata.json`：`state`、`reference`、`previous_wrench`、`target_wrench` 和时间戳按样本对齐；metadata 保存源文件 SHA-256、样本数、拒绝原因和完整契约。

训练/验证/测试按完整 run 或路线划分，避免相邻帧同时出现在训练和验证中。`--mode overfit` 只用于工程链检查，不能当作泛化实验。

### 4.3 BC loss 和优化

BC 的主损失是归一化广义力 MSE：

```text
e = (τ_pred - τ_teacher) / wrench_scale
L_wrench = mean(e²)
```

正式训练还加入内部动作幅值正则：

```text
L_effort = mean(q²)
L = L_wrench + λ_effort L_effort
```

默认 `λ_effort = 0.01`，优化器为 AdamW，学习率 `3e-4`，weight decay `1e-4`，梯度范数裁剪为 1.0。过拟合工程模式关闭 effort 正则，避免把“能否记住 128 个样本”和“动作偏好”混在一起。验证同时报告 Fx/Fy/Nz 的 RMSE、MAE 和归一化动作 effort。

仓库计划记录过一次 P0 工程链结果：单个 run 导出 576 个样本，取 128 个样本过拟合，最佳 checkpoint 记录为 epoch 1474，训练样本 RMSE 记录为 `Fx=0.130 N`、`Fy=0.060 N`、`Nz=0.00476 N·m`，重载 checkpoint 后输出逐值一致。该数字是已有计划记录，不能替代独立路线验证。

### 4.4 BC 阶段暴露的问题与修复

| 问题 | 修复 |
|---|---|
| 直接回归任意广义力可能生成推进器不可达命令 | 改为 `tanh → 推进器范围 → B_h` 的有界隐式推进器头 |
| 参考窗口或 odometry 时间错位造成伪标签 | 导出器按动作时间做因果匹配，并记录拒绝原因 |
| 角度在 `π` 附近不连续 | 输入使用 `sin(Δyaw)`、`cos(Δyaw)` |
| 相邻帧随机拆分导致验证泄漏 | 以完整 run/路线划分数据集 |
| 网络输出在训练后漂移到非法数值 | dataset、loss、推理和 ROS 节点都检查 finite；异常周期安全停机 |

## 5. 阶段二：DAgger

### 5.1 为什么从 BC 进入 DAgger

BC 只看 teacher 访问的状态；一旦策略闭环执行出现小误差，状态会逐渐离开 teacher 轨迹，误差可能累积。这是典型的 covariate shift。DAgger 的目标是让 MPC 在**当前策略实际访问的状态**上重新提供标签。

采集入口是 [`collect_dagger.py`](src/hofa_policy/src/hofa_policy/collect_dagger.py)：Stonefish 闭环由 BC/当前策略控制，MPC 同步输出 teacher 广义力；两者同时写入 run。DAgger 数据不是把策略动作当标签，而是保留策略访问状态和 MPC teacher 标签。

### 5.2 DAgger 数据的因果组织

[`export_dagger.py`](src/hofa_policy/src/hofa_policy/export_dagger.py) 读取：

```text
odometry.csv
mpc_reference_window.csv
dagger_teacher_generalized_force.csv
generalized_force.csv              # 当前策略实际发布的动作
controller_status.csv
```

关键点是 `previous_wrench` 使用**上一时刻策略发布的广义力**，而不是上一时刻 teacher 力。这样训练输入和部署时的历史动作语义一致。第一帧没有历史策略动作时使用零向量，并在 metadata 中记录计数。时间戳、odometry 年龄和参考窗口年龄仍执行因果检查。

### 5.3 DAgger 的多步训练 loss

[`dagger_dataset.py`](src/hofa_policy/src/hofa_policy/dagger_dataset.py) 将每条 run 组织成长度为 `rollout_steps` 的连续序列，默认长度为 3。[`train_dagger.py`](src/hofa_policy/src/hofa_policy/train_dagger.py) 在每个序列上滚动预测：

```text
step 0: previous = 数据中的上一策略动作
step k: previous =
        teacher target[k-1]，概率 p
        上一步策略预测（detach），概率 1-p
```

`p` 从默认 0.8 线性衰减到 0.2，这就是 teacher forcing 到 policy rollout 的过渡。每一步使用归一化广义力误差，默认轴权重为 `[Fx:1.5, Fy:1.0, Nz:0.8]`：

```text
L_DAgger = mean_k mean( [1.5, 1.0, 0.8] ⊙
                        ((τ_pred[k] - τ_teacher[k]) / wrench_scale)² )
```

验证阶段完全使用模型自己的上一预测，不使用 teacher 动作，从而检查多步闭环误差而非单帧拟合。DAgger 可以从 BC checkpoint 初始化，也可以从默认网络开始。

### 5.4 DAgger 的验收重点

不能只看离线 MSE，应至少同时检查：

- 策略访问状态上的 teacher-conditioned 误差；
- free-running 多步误差是否随时间漂移；
- 急弯、hairpin、switchback 等路线的闭环成功率；
- 广义力和候选推进器是否有越界；
- 推理 P95/P99、输入年龄和安全停机次数。

## 6. 部署时的约束、饱和与安全链

ROS 控制节点是 [`hofa_policy_controller_node.py`](scripts/hofa_policy_controller_node.py)。它把“网络输出受限”和“真实执行器可能饱和”分成两层处理。

### 6.1 网络层的硬保证

对 BC/DAgger 主网络：

1. `q = tanh(raw_action)`，数学上满足 `-1 ≤ q_i ≤ 1`。
2. `f_i = f_mid_i + f_half_i q_i`，因此每一路候选水平推进器力满足 `force_min_i ≤ f_i ≤ force_max_i`。
3. `τ = B_h f`，候选广义力必然来自四推进器可达集合的线性像，而不是任意三维向量。
4. ROS 节点再次检查 finite、推进器上下限和 `B_h f` 与待发布 wrench 的一致性；不一致时不发布该动作。

这保证的是**候选水平四推进器参数化的输入约束**。它不等价于“真实六推进器分配器永远不饱和”，因为下游还有六推进器几何、垂向力耦合、PWM 曲线和死区。

### 6.2 RL 残差层的硬保证

RL 动作是归一化的三维残差 `a ∈ [-1,1]^3`。组合逻辑在 [`residual_core.py`](src/hofa_policy/src/hofa_policy/residual_core.py)：

```text
desired_residual = clip(a, -1, 1) ⊙ residual_limit
residual_t = clip(desired_residual,
                  residual_{t-1} - slew,
                  residual_{t-1} + slew)
command_t = base_wrench_t + residual_t
```

当前配置：

```text
residual_limit       = [1.0 N, 0.4 N, 0.08 N·m]
residual_slew/step   = [0.3 N, 0.15 N, 0.025 N·m]
```

因此 RL 不能一步跳出残差幅值，也不能产生超过每步 slew 的突变。零动作、零历史残差时 `command == base_wrench`，零残差严格等价于冻结 DAgger 基线。ROS 残差部署还通过伪逆得到候选推进器力、按物理范围 clip，再用 `B_h` 重建最终 wrench，保持候选力和 wrench 一致；如果六推进器分配已经饱和，重建后的实际 wrench 可能与未饱和的 `base_wrench + residual` 不同，这个差异会被记录并进入评测。

### 6.3 ROS 周期级安全保护

控制节点在推理前检查：

- controller gate 是否允许控制；
- odometry 和参考窗口是否存在、是否新鲜；
- 默认运行时 odometry/reference 年龄上限为 0.20/0.30 s（离线导出阈值更严格，为 0.05/0.15 s）；
- 位置、世界速度和偏航角速度是否在状态包络内；
- 输入、输出和候选推进器是否 finite。

任何失败都清零历史、发布零水平广义力并进入 `SAFETY_STOP` 或等待状态。只有完整通过检查的周期才更新 `previous_wrench`。这避免了“异常输出被当作下一周期历史动作”继续传播。

### 6.4 真实执行器饱和如何处理

RL 环境使用 [`rl_actuation.py`](src/hofa_policy/src/hofa_policy/rl_actuation.py) 复用实际六推进器 allocator：

1. 用实际伪逆把水平/深度 wrench 转成六路 proposed force。
2. 按 `[-6×9.80665, 6.9×9.80665]` 的物理范围 clip。
3. 通过实际 PWM 曲线反算 realized force，包含启动死区。
4. 由 realized force 重建实际广义力，并记录 `saturated` 标志。

所以训练和评测看到的是“发送命令”和“物理实际实现”的区别；饱和率、残差 RMS、残差限幅命中率会写入 episode metrics。最终判断不能只看网络输出范围，还要看实际 allocator 的饱和和 realized wrench。

## 7. 阶段三：冻结 DAgger 上的 RL 残差微调

### 7.1 为什么采用残差而不是直接 RL

绝对 RL 需要同时学会参考跟踪、动力学补偿、推进器分配和安全边界，探索空间大且容易破坏已有策略。当前方案冻结 `train_dagger_v1/best.pt`，RL 只学习小幅修正：

```text
base_wrench = frozen DAgger policy(state, reference, previous_wrench)
command     = base_wrench + bounded_residual
```

冻结网络的参数哈希在训练前后检查；RL 优化不会反向更新 DAgger 主网络。

### 7.2 RL 观测

RL 观测契约为 212 维：

```text
state / scale                         3
reference / scale                    20×10 = 200
previous commanded wrench              3
frozen base wrench                     3
previous applied residual              3
                                      ---
                                      212
```

该观测和 ROS 残差部署共用，避免训练时看到部署时不存在的变量。动作空间是 `Box([-1,-1,-1], [1,1,1])`。

### 7.3 物理环境和 episode

[`residual_env.py`](src/hofa_policy/src/hofa_policy/residual_env.py) 通过 fixed-step Stonefish bridge 推进真实 1.6 物理引擎：

- physics 频率 100 Hz，控制周期 0.1 s；
- simulator 只在 `reset/step` 推进，优化器更新期间不会偷偷前进；
- 训练路线为 `right_angle`、`hairpin`、`switchback` 的长度/镜像变化；
- 深度继续由 PID 保持；
- 训练、验证、测试使用不重叠的固定随机种子。

场景 XML、桥接程序、冻结 checkpoint、配置和关键源码都会写入 provenance/hash，便于判断不同物理版本是否被误混。

### 7.4 RL reward

奖励在 [`residual_reward.py`](src/hofa_policy/src/hofa_policy/residual_reward.py) 中拆成可审计项：

```text
r = progress
    - w_cross  · normalized_cross_track² · dt
    - w_yaw    · (1 - cos(yaw_error)) / yaw_denominator · dt
    - w_rate   · normalized_yaw_rate² · dt
    - w_lag    · normalized_lag² · dt
    - w_effort · mean((residual / magnitude_scale)²) · dt
    - w_smooth · mean(((residual - previous_residual) / slew_scale)²) · dt
    - w_time · dt
    + success_bonus - failure_penalty
```

默认权重为：progress 5.0、cross-track 2.0、yaw 1.0、yaw-rate 0.3、lag 0.2、effort 0.03、smooth 0.02、time 0.05、success/failure 20.0。

奖励有两套尺度：legacy 和 tolerance。tolerance 方案把横向误差按 1 cm、航向误差按 5°、角速度按 0.5 rad/s、lag 按 1 m 归一化；这些是消融实验起点，不是已经证明的工程阈值。残差正则的分母固定为物理残差上限和 slew 上限，不能通过改分母掩盖大动作。

### 7.5 TD3/SAC 配置

两种算法共用环境和配置：

| 项目 | 默认值 |
|---|---:|
| actor/critic 网络 | `[128, 128]` |
| learning rate | `3e-4` |
| replay buffer | `200000` |
| learning starts | `2000` |
| batch size | `256` |
| gamma / tau | `0.99 / 0.005` |
| TD3 action noise | `0.1` |
| SAC entropy | `auto_0.02` |

TD3 的初始确定性残差为零；SAC 的初始均值为零且方差较小。消融实验另提供前 2000 步的零中心截断高斯 warmup：`std=0.05`、clip `±0.15`，用于减少一开始在整个残差范围内均匀探索对基线的破坏。

## 8. 验证、模型选择和“RL 是否真的有用”

RL 不能按训练 return 单独选模型。[`residual_validation.py`](src/hofa_policy/src/hofa_policy/residual_validation.py) 在每次验证同时运行：

1. 零残差 DAgger 基线；
2. 当前 TD3/SAC 候选；
3. 相同路线、相同案例和相同随机种子的 paired episode。

候选只有在以下共同门槛全部满足时才允许击败零残差：

- 成功率不能下降；
- yaw、转弯 yaw、转弯角速度 RMSE 相对退化不超过 5%；
- 平均终点距离不增加超过 5 mm；
- 平均完成时间不增加超过 0.5 s；
- 平均横向 RMSE 至少改善 0.1 mm。

因此 `best.zip` 是“通过物理门槛后横向误差最小”的候选；如果没有候选通过，选择结果是 `zero_residual`。候选即使被保存，也不能把 `best.zip` 的存在解释成已经超过 DAgger。

仓库的 RL 说明记录过：上一版 residual RL 独立评测没有整体超过零残差 DAgger。因此下一轮采用固定网络、固定残差边界、固定正则和固定预算的 2×2 消融，只改变 warmup 采样和 tracking reward 尺度：

| 组别 | warmup | tracking reward |
|---|---|---|
| A | 原 SB3 全范围均匀动作 | legacy |
| B | 小幅零中心高斯 | legacy |
| C | 原 SB3 全范围均匀动作 | tolerance |
| D | 小幅零中心高斯 | tolerance |

这样可以区分“探索一开始破坏基线”和“误差尺度不适合学习”两个原因，而不把网络变大或训练预算拉长作为第一反应。

## 9. 典型问题、定位和解决办法

| 现象/风险 | 根因 | 当前处理 |
|---|---|---|
| BC 离线误差小，闭环误差累积 | covariate shift，策略访问了 teacher 没覆盖的状态 | DAgger 在策略闭环状态上重新采 MPC 标签；序列 rollout 逐步减少 teacher forcing |
| 历史动作语义错位 | 训练把上一 teacher 动作当 history，部署却是上一策略动作 | DAgger 导出明确使用上一策略发布的 wrench |
| 网络输出不可达或下游分配异常 | 直接回归 3 维 wrench 没有推进器可行性结构 | 4 维 `tanh` 隐变量、固定 force bounds 和 `B_h`；推理末端再做一致性检查 |
| 残差一步跳变导致振荡 | RL 动作虽然归一化，但变化率没有物理约束 | `residual_limit + residual_slew_per_step` 双重限制；reward 还有 smooth 正则 |
| residual RL 训练 return 上升但闭环没有改善 | reward 可能被进度/终止项主导，训练目标和比较目标不一致 | paired zero-residual validation、共同 comparison reward、物理 gate 和保守选择规则 |
| RL 训练时仿真时间与 optimizer 更新耦合 | 在线仿真推进和学习更新混在一起导致不可复现 | fixed-step Stonefish bridge；只在 reset/step 推进；smoke 检查暂停时钟不前进 |
| hairpin 交叉点导致参考进度跳到后段 | 最近点投影可能跨到另一条分支 | `rl_reference.py` 只允许当前 arc 附近的相邻线段；测试明确检查不能跳段 |
| PWM 死区和理论 wrench 不一致 | 只记录 command force，没有记录实际 PWM 实现 | 复用真实 allocator、PWM 曲线和死区，记录 proposed/realized wrench 和 saturation |
| stale odometry/reference 让策略使用过期输入 | ROS 回调与控制周期不同步 | 年龄阈值、未来时间戳检查、失败周期归零并清历史 |
| 单次 smoke 成功被误当成策略有效 | smoke 只验证 Gym/API、固定步进和基线冻结 | 独立验证、固定种子、多 seed paired evaluation；文档明确 smoke 不是性能验收 |

## 10. 数据、代码和产物组织

### BC

```text
results/.../run_*/
  odometry.csv
  mpc_reference_window.csv
  generalized_force.csv
  controller_status.csv
        │ export_run.py
        ▼
results/.../datasets/<run>/
  samples.npz
  metadata.json
        │ PolicyDataset + train.py
        ▼
  best.pt / last.pt / run.json / learning_curve.csv
```

### DAgger

```text
results/.../raw/<trajectory>/run_*/
  odometry.csv
  mpc_reference_window.csv
  dagger_teacher_generalized_force.csv
  generalized_force.csv              # policy command
  controller_status.csv
        │ export_dagger.py
        ▼
results/.../datasets/<trajectory>/
  samples.npz
  metadata.json
        │ DaggerSequenceDataset + train_dagger.py
        ▼
  best.pt / last.pt / run.json / learning_curve.csv
```

### Residual RL

```text
results/.../residual_rl/<algo_seed>/
  training.json / completed.json
  train_env/provenance.json
  validation_env/
  checkpoints/
  final.zip / best.zip
  validation.jsonl / selection.json
  episodes.jsonl / TensorBoard
```

训练环境还会保存扁平化 scenario、物理源文件哈希、bridge 哈希、冻结 DAgger checkpoint 哈希和关键源码哈希。这样可以区分“算法变化”和“物理/场景变化”。

## 11. 当前状态和验收边界

| 阶段 | 当前判断 | 说明 |
|---|---|---|
| BC 契约、模型、导出、训练入口 | **已实现** | 代码、配置、dataset schema 和 checkpoint contract 齐全 |
| BC 小样本过拟合链路 | **已记录** | `PLAN.md` 记录过 576 样本导出和 128 样本过拟合结果 |
| 多路线 teacher 采集 | **脚本已实现，正式覆盖待验收** | 采集器、轨迹套件和 headless 场景已存在 |
| DAgger 因果导出与序列训练 | **已实现** | teacher label、上一策略动作和 rollout loss 已分开定义 |
| ROS Shadow/安全检查 | **已实现，系统级复现待验收** | 有 Shadow 产物和输入年龄/候选力检查 |
| 固定步进 Stonefish residual 环境 | **已实现** | smoke 会检查 API、可复现性、时钟和冻结 baseline |
| TD3/SAC residual 训练 | **已实现，效果待验收** | 有统一配置、验证 callback、paired gate 和并行入口 |
| residual RL 超过 DAgger | **不能默认宣称** | 现有说明明确记录上一版没有整体超过零残差 DAgger |
| MCU/Orin 与当前分配矩阵完全一致 | **待验收** | 需要硬件/目标端确认顺序、符号、限幅和 PWM 标定 |

因此，推荐的正式验收顺序仍是：

```text
BC 小样本过拟合
→ 按路线划分的 BC 离线验证
→ DAgger 闭环采集与多步验证
→ Shadow（不接管车辆）
→ Stonefish 闭环
→ Orin 推理延迟
→ MCU 分配一致性
→ residual TD3/SAC paired validation
```

## 12. 关键入口索引

- 契约和几何：[`contract.py`](src/hofa_policy/src/hofa_policy/contract.py)
- 特征构造：[`features.py`](src/hofa_policy/src/hofa_policy/features.py)
- BC 模型：[`model.py`](src/hofa_policy/src/hofa_policy/model.py)
- BC 数据导出：[`export_run.py`](src/hofa_policy/src/hofa_policy/export_run.py)
- BC dataset/training：[`dataset.py`](src/hofa_policy/src/hofa_policy/dataset.py)、[`train.py`](src/hofa_policy/src/hofa_policy/train.py)
- DAgger 导出/序列训练：[`export_dagger.py`](src/hofa_policy/src/hofa_policy/export_dagger.py)、[`dagger_dataset.py`](src/hofa_policy/src/hofa_policy/dagger_dataset.py)、[`train_dagger.py`](src/hofa_policy/src/hofa_policy/train_dagger.py)
- DAgger 采集：[`collect_dagger.py`](src/hofa_policy/src/hofa_policy/collect_dagger.py)
- ROS 部署安全链：[`hofa_policy_controller_node.py`](src/hofa_policy/scripts/hofa_policy_controller_node.py)
- RL 观测/残差合成：[`residual_core.py`](src/hofa_policy/src/hofa_policy/residual_core.py)
- RL 环境：[`residual_env.py`](src/hofa_policy/src/hofa_policy/residual_env.py)
- RL reward：[`residual_reward.py`](src/hofa_policy/src/hofa_policy/residual_reward.py)
- RL 训练：[`residual_train.py`](src/hofa_policy/src/hofa_policy/residual_train.py)
- RL paired validation：[`residual_validation.py`](src/hofa_policy/src/hofa_policy/residual_validation.py)
- 实际六推进器 allocator：[`rl_actuation.py`](src/hofa_policy/src/hofa_policy/rl_actuation.py)
- 主配置：[`policy.yaml`](src/hofa_policy/config/policy.yaml)、[`residual_rl.json`](src/hofa_policy/config/residual_rl.json)
- RL 消融计划：[`residual_ablation.json`](src/hofa_policy/config/residual_ablation.json)
- 已有 RL 运行说明：[`RESIDUAL_RL.md`](src/hofa_policy/RESIDUAL_RL.md)

这份文档描述的是当前实现和已记录的实验边界。凡是标为“待验收”的内容，仍需要在固定场景、固定 seed、相同参考限速和真实分配器链路下补齐证据后，才能写成性能结论。
