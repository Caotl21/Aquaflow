# HOFA Policy 实施计划

## 目标

`hofa_policy` 用一个状态与参考轨迹策略网络替代水平面控制链中的 Layer 1、Layer 2 和 HOFA inverse 在线计算，同时保持现有 ROS/MCU 接口不变：

```text
车辆状态 + 20 步参考窗口 + 上一周期广义力
                    ↓
             hofa_policy
                    ↓
内部有界推进器变量 q ∈ [-1,1]^4
                    ↓
候选水平推力 f = f_mid + f_half ⊙ q
                    ↓
水平广义力 τ = B_h f
                    ↓
/controller/generalized_force (WrenchStamped)
                    ↓
MCU 固件分配 T1-T4 并输出 PWM
```

网络内部的 `f` 是可行性见证，不下发给 MCU。实际外部输出仍是 `[X,Y,N]` 广义力。

## 固定边界

- 第一版只替换水平面 3-DOF 控制，深度控制与垂向推进器保持现状。
- 网络、特征和训练标签采用 NED/FRD 约定，与 `/controller/generalized_force` 和 MCU 接口一致。
- `B_h`、推进器上下限、坐标变换和 ROS 消息格式使用解析代码，不交给网络拟合。
- 第一版不使用相机。`aquaflow_bc` 属于规划层，`hofa_policy` 不依赖也不修改它。
- MPC 在数据采集、Shadow 和回退阶段继续作为 teacher。

## 数据契约 v1

每个样本对应一个控制周期。

### 输入

1. 当前机体系速度 `state = [u, v, r]`。
2. 20 点参考窗口，每点 10 维：
   `[x_body, y_body, sin(delta_yaw), cos(delta_yaw), vx_body, vy_body, dpsi, ax_body, ay_body, ddpsi]`。
3. 上一周期已发布的水平广义力 `previous_wrench = [Fx, Fy, Nz]`。

### 标签

当前 HOFA-MPC 最终发布的水平广义力：

```text
target_wrench = [Fx, Fy, Nz]
```

标签已经包含 Layer 1、Layer 2、HOFA inverse、广义力缩放、安全限幅和 FLU→FRD 转换的最终行为。

### 样本接收条件

- 控制器状态为 `ACTIVE`。
- MPC 求解成功。
- 状态时间戳不晚于动作且年龄不超过 50 ms。
- 参考窗口不晚于动作且年龄不超过 150 ms。
- 参考窗口包含连续的 20 个索引。
- 数据集按完整 run/路线划分 train、val、test，不随机拆相邻帧。

## 模型 v1

- 参考窗口由逐点 MLP 和轻量 1D 卷积编码。
- 状态与上一广义力由独立 MLP 编码。
- 融合后输出 4 维 `tanh` 动作。
- 解析执行器层把动作映射为推进器力，再通过 `B_h` 合成可达广义力。
- 主损失为归一化广义力 MSE；辅损失约束内部推进器动作幅值。

## 阶段

### P0：数据和工程链

- [x] 建立独立 catkin/Python 包。
- [x] 固化特征、动作、坐标系与推进器几何契约。
- [x] 实现已有 MPC 实验目录到策略数据集的因果转换。
- [x] 实现 BC 模型、训练入口和基础测试。
- [x] 用一个 run 导出 576 个因果样本。
- [x] 完成 128 样本过拟合并核对 checkpoint 重载。最佳 checkpoint 位于 epoch 1474；训练样本 RMSE 为 `Fx=0.130 N`、`Fy=0.060 N`、`Nz=0.00476 N·m`，权重重载输出逐值一致。

### P1：多路线 BC

- [x] 实现无障碍多轨迹套件生成器。
- [x] 实现 Stonefish HOFA-MPC teacher 批量采集与自动数据导出。
- [x] 分离 `/hofa_policy/teacher_global_path` 与 Aquaflow 规划话题。
- [x] 增加与原车辆动力学一致的 camera-free headless 控制采集场景。
- [ ] 完成一条 smoke teacher run。
- [ ] 采集不同直线、弯道、速度和初始误差的完整 teacher runs。
- [ ] 按路线划分 train/val/test。
- [ ] 报告广义力 MAE/RMSE、饱和比例、推理 P95/P99。
- [ ] 检查网络内部合成的广义力始终位于推进器可达集合。

### P2：Shadow

- [x] 新增 ROS Shadow 节点，只发布诊断话题，不控制车辆。
- [x] 对齐网络与 MPC 输出，记录误差、延迟、候选推力和输入年龄。
- [x] 新增逐周期 CSV 与 Shadow 汇总报告生成器。
- [ ] 确认 Orin 与 MCU 的 `B_h`、推进器顺序、符号和限幅一致。

### P3：闭环与回退

- [ ] 增加 `controller:=policy` 启动模式。
- [ ] 保留状态安全检查、Wrench 硬限幅和 MPC 回退。
- [ ] 比较到达率、跟踪误差、超调、控制平滑度和回退比例。
- [ ] 运行 DAgger，收集策略访问状态上的 MPC 标签。

### P4：RL 微调

- [ ] 在 BC/DAgger checkpoint 上进行残差或策略微调。
- [ ] 随机化质量、阻尼、水流、延迟和推进器降额。
- [ ] RL 不得绕过有界推进器参数化、安全监督和 MPC 回退。

## 验收顺序

1. 小样本过拟合。
2. 独立路线离线评估。
3. Shadow。
4. Stonefish 闭环。
5. Orin 推理基准。
6. MCU 联调。
7. DAgger/RL。
