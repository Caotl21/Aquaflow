# HOFA-MPC 算法与实现

BricsBot（4 推进器水平全驱 AUV）平面 3-DOF 轨迹跟踪控制器，运行在 ROS1 Noetic + Stonefish 仿真环境。

核心是**两层结构**：Layer 1 把执行器物理限制折算成"虚拟加速度"的可行盒子，Layer 2 在这个盒子里解一个线性误差 MPC，最后用 HOFA 逆变换把加速度还原成机体广义力。

```
全局路径 (privileged_teacher)
      │
      ▼  arc_profile.build_arc_profile  —— 弧长重参数化 + 速度/时间剖面
reference_processor  —— 绝对计划时钟 s_desired(t) 锚定，输出时间索引窗口
      │  /controller/reference_trajectory_window  (20 点 × 0.1 s)
      ▼
hofa_mpc_controller_node  (10 Hz)
      ├─ Layer 1  constraints.SafeInnerBoxStrategy  —— LP 验证的虚拟输入盒 × Np
      ├─ Layer 2  mpc.HofaMPC                       —— L-BFGS-B 解线性误差 MPC
      └─ hofa.hofa_inverse                          —— a_c → τ = [X, Y, N]
      │  /controller/generalized_force  (WrenchStamped, NED/FRD)
      ▼
brics6_thruster_allocator  —— 分配到 T1~T4
```

---

## 1. 建模基础

### 1.1 3-DOF 平面模型 (`model.py`)

状态 `η = [x, y, ψ]`（世界系）、`ν = [u, v, r]`（机体系）：

```
M ν̇ + C(ν)ν + D(ν)ν = τ + d
η̇ = J(ψ) ν
```

| 项 | 说明 |
| --- | --- |
| `M = diag(m_x, m_y, I_z)` | 有效惯性 = `dry_mass + added_mass`，见 `vehicle_params.py` |
| `D(ν) = diag(d_l,i + d_q,i·\|ν_i\|)` | 线性 + 二次阻尼，对角 |
| `C(ν)` | 标准 3-DOF 反对称形式，默认关闭（`coriolis_enabled: false`） |
| `J(ψ)` | 机体 → 世界旋转矩阵；`J̇(ψ, r)` 为其时间导数 |

`rk4_step()` 提供离线仿真用的四阶龙格库塔积分。

### 1.2 惯性参数的分离存储 (`vehicle_params.py`)

控制器需要的是**有效惯性**（刚体质量 + 附加质量），但这两半在不同平台上行为完全不同，所以配置里刻意分开存：

- `dry_mass` 是车的属性，仿真和实物相同，用秤称出来。
- `added_mass` 是车**和流体模型**的属性。Stonefish 用包围碰撞网格的最小体积椭球推导附加质量，对开放框架壳体来说这个椭球包住的水远多于实际排水量 —— 4.29 L 实体装在 34 L 包络里，而实际排水 7.775 L，椭球约为排水体积的 12 倍。结果就是 `vehicle_sim.yaml` 里 `added_mass: [233.06, 286.06, 1.88]`，比真实值高约 20 倍。

因此 **`vehicle_sim.yaml` 和 `vehicle_real.yaml` 之间绝对不能互抄数值**。`vehicle_real.yaml` 目前把 `added_mass` 留为 0（"尚未测量"），因为低估惯性只会让车迟钝、积分器能补回来；高估惯性会让每次变速都把推进器打满。

`resolve_inertia()` 同时兼容旧的 9 元素 `mass_matrix` 扁平形式，但那种形式里附加质量已经折进去、无法归因。

### 1.3 HOFA 变换 (`hofa.py`)

HOFA = High-Order Forward Attachment，本质是**反馈线性化**：把"世界系加速度"提升为虚拟输入。

**正向**（状态 + 力 → 世界加速度）：

```
η̈ = f(x) + G(x)·τ
f(x) = J̇ν + J M⁻¹(−Cν − Dν)
G(x) = J M⁻¹
```

**逆向**（世界加速度 → 机体广义力），控制器实际使用的一步：

```
τ_d = M Jᵀ (a_c − J̇ν) + C(ν)ν + D(ν)ν
```

这一步把科氏力、阻尼、旋转坐标系的离心项一次性全部前馈补偿掉，于是 Layer 2 面对的就是一个纯双积分器。

`hofa_forward_inverse_identity()` 是自检函数，验证 `forward(inverse(a_c)) ≈ a_c`。

---

## 2. Layer 1：虚拟输入约束 (`constraints.py`)

**问题**：MPC 的决策变量是世界系加速度，但真实限制在推进器推力上。

**做法**：把推力盒 `[f_min, f_max]⁴` 通过 `G = J M⁻¹ B_h` 映射到加速度空间得到可达集（一个 zonotope），再在里面取一个保守的轴对齐内接盒。

`SafeInnerBoxStrategy._build_verified_box()` 的算法：

1. 中心 `c = f_drift + G·f_mid`，半宽 `h_raw = Σ|G|·f_half`（这是**外接**盒，不是内接）。
2. 按 `safe_box_scale`（默认 `1/3`）缩放。
3. **LP 验证**：对缩放后盒子的 8 个角点各解一个 `scipy.optimize.linprog`（`A_eq = G`，`b_eq = corner − f_drift`，变量边界为推力限幅），确认每个角点都真的可达。
4. 若不可达，对 scale 做 24 次**二分搜索**，退回到最大可行缩放。

这保证 MPC 的可行域是真实可达集的**子集** —— 解出来的加速度一定分配得出去。代价是每步 8 次 LP，所以 `layer1_time_ms` 在 `ControllerStatus` 里单独计时上报。

控制节点每周期对**整个 horizon 的每一步**调 `compute_for_step()`（输入是 `predict_nominal_states()` 用上周期解左移后预测的名义状态），得到 Np 个盒子后**冻结**，再进 Layer 2 求解。这样避免了"约束依赖轨迹、轨迹依赖约束"的迭代。

`CurrentStateBoxStrategy` 是只基于当前状态、穷举 8 个推力角点的简化版本，保留作调试/兜底。

---

## 3. Layer 2：误差 MPC (`mpc.py`)

### 3.1 预测模型

误差状态 `z = [e_η; ė_η] ∈ ℝ⁶`，决策变量是**修正量** `w`（总虚拟加速度 = 参考加速度 + `w`）。经 HOFA 线性化后模型就是双积分器：

```
Ad = [[I₃, dt·I₃], [0, I₃]]
Bd = [[0.5·dt²·I₃], [dt·I₃]]
z[i+1] = Ad·z[i] + Bd·w[i] + d[i]
```

`d[i] = Ad·ref[i] − ref[i+1]` 是**时变参考引起的仿射项**（`_reference_affine_terms`）。yaw 分量单独处理：先加上预期的 `dt·dψ` 增量再 `wrap_to_pi`，避免 ±π 跳变被当成巨大误差。

### 3.2 代价函数

```
J = Σ_{i=0}^{Np−1} [ zᵢᵀ Qᵢ zᵢ + wᵢᵀ R wᵢ + Δwᵢᵀ S Δwᵢ ]
Δw₀ = w₀,  Δwᵢ = wᵢ − wᵢ₋₁
Q_{Np−1} = F = Q × terminal_multiplier
```

权重（`config/mpc.yaml`）：

| 权重 | 值 |
| --- | --- |
| `pose` | `[20, 20, 20]` |
| `pose_rate` | `[10, 10, 8]` |
| `virtual_input` (R) | `[0.20, 0.20, 0.14]` |
| `input_increment` (S) | `[1.20, 1.20, 0.75]` |
| `terminal_multiplier` | `4.0` |

> **`pose_rate` 从 4.0 提到 10.0 的原因**：速度权重只有位置权重的 1/5 时，求解器拿速度误差换位置误差，速度环欠阻尼 —— 对着 0.2 m/s 的参考在 0.21~0.34 m/s 之间以约 10 s 的周期振荡。偏航角速率权重留得比两个平移轴略低。

### 3.3 求解

- **求解器**：`scipy.optimize.minimize`，`L-BFGS-B`，`Np × 3 = 60` 个变量。
- **箱式约束**来自 Layer 1，但要做平移：决策变量是修正量，所以 `lb[i] = bounds[i].lower − ref[i].acceleration`。
- **解析梯度**：`_cost_and_grad()` 用一次前向 rollout + 一次反向伴随（adjoint）传播算精确梯度，替代原先 O(n_var) 次有限差分的代价评估。这是性能关键优化。
- **Warm start**：上周期解左移一步（`previous_input_sequence()`）。

**成功判定**（`status ∈ {0, 1}`）：

> 滚动时域控制器可以直接用 maxiter 的迭代结果 —— 它是一个可行的次优控制，下一周期还会从它 warm start。把 `status == 1` 判成失败会让每次迭代受限的求解都看起来像故障。只有 `status == 2`（异常终止，如线搜索失败）才意味着结果不可用。

这就是 git 历史里 "mpc求解器死锁修正" 那一条。`status = −1` 表示求解器本身抛了异常。

### 3.4 时间参数

`control_rate_hz: 10.0`，`horizon: 20` → **2.0 s 预览**。

> **为什么必须是 2 s**：以 `max_decel = 0.12 m/s²` 从 0.30 减速到 0.13 m/s（应对 R = 0.38 m 的弯）需要 0.94 s。预览只有 0.5 s 时弯道在需要刹车的时刻还"看不见"，实测车跑宽 0.83 m。把 `mpc_dt_s` 从 0.05 加宽到 0.1 不增加变量数（求解时间不变），却把预览翻倍。

---

## 4. 推力分配 (`allocator.py`)

`τ = B_h · f_h`，`B_h` 第 i 列为 `[d_x, d_y, r_x·d_y − r_y·d_x]`。

BricsBot 构型：4 个 45° 斜置推进器，位于 `(±0.12755, ±0.12755)`，单推力范围 `[−58.84, +67.67] N`。

分配用**加权伪逆 + 限幅**：

```
W = diag(1, 1, yaw_weight=2)
B⁺ = (BᵀWᵀB + λI)⁻¹ BᵀWᵀ,   λ = 1e-4
f_h = clip(B⁺ τ, f_min, f_max)
```

返回残差 `‖B_h·f_h − τ‖`。这里不解 QP —— 限幅后残差小是由 Layer 1 的可达性保证的。

注意：`hofa_mpc_controller_node` **只发布广义力**，实际分配由 `aquaflow_stonefish` 包的 `brics6_thruster_allocator.py` 完成。本包的 allocator 主要作用是给 Layer 1 提供 `B_h` 和推力限。

---

## 5. 参考生成

这部分承载了最多的工程经验，也是改动最集中的地方。

### 5.1 弧长剖面 `arc_profile.build_arc_profile()`

**纯函数，无 ROS 依赖，无节点状态**。它是"控制器被要求跟随的计划"的**唯一定义** —— `reference_processor` 在线用它构造参考窗口，`run_mpc_experiment` 离线用它对一次运行评分。保持单一实现，是评测既独立于控制器、又衡量同一份意图的前提。

流程：

1. **航向**：从几何切线算 `yaw[i] = atan2(Δy, Δx)`，**不信任输入 pose 的姿态** —— 全局路线的切线才是权威航向。随后按弧长做滑动平均平滑（在 s 坐标下滤波，避免点间距不均造成曲率尖刺）。
2. **曲率**：有符号，`κ[i] = wrap_to_pi(ψ[i+1] − ψ[i−1]) / ds`。用 `|dψ|/ds` 会在角度回绕处高估曲率，而且分不出左右转。
3. **速度上限**：
   - 曲率限：`min(max_speed, max_yaw_rate / |κ|)`
   - **偏航角加速度限**：`v ≤ sqrt(remaining_yaw_accel / |dκ/ds|)`，其中
     `remaining = max(max_yaw_accel − |κ|·a_lon, reserve_floor)`

   > `yaw_accel_reserve_ratio = 0.2` 的地板是一个 bug 修复。被减项为"弯中变速预留偏航角加速度预算"，但它假设各处都是最坏情况的纵向加速度 —— 包括 dv/dt ≈ 0 的稳态巡航。在急弯上这个假设把预算压成负数、速度上限钉死在精确的 0，也就是**计划要求在路线中间急停**。有了地板，约束只会退化而不会崩溃。

4. **加减速平滑**：前向扫描（`max_accel`）+ 反向扫描（`max_decel`），终点速度置 0。这给出物理连续的速度剖面，而不是各点独立变速。
5. **到达时刻表**：`time[i] = time[i−1] + ds / max(min_profile_speed, 平均速度)`。`min_profile_speed = 0.08` 防止终点静止导致最后一段时间无穷大。
6. `initial_speed` 参数：`None` 表示起点不约束（可能从全速开始）；`0.0` 强制从静止起步 —— **离线评分用 0.0**，因为那才是车真能从站定状态达成的计划。

返回 `dict`：`x, y, z, yaw, curvature, speed, s, time, total_length`。

### 5.2 锚点策略：`schedule` vs `projection`

这是 `reference_processor` 最关键的设计决策。

| 模式 | 行为 | 后果 |
| --- | --- | --- |
| `projection`（旧） | 每周期把参考重锚到车自身在路径上的投影 | **纵向误差恒等于 0**，下游控制器根本看不见自己落后多少 |
| `schedule`（默认） | 维护绝对计划时钟，`s_desired(t) = interp(elapsed, time, s)` | 控制器能看到真实的沿程误差 |

`schedule` 模式的两个细节：

- **领先限幅 + 时钟回滚**：`max_schedule_lead_m = 1.0` 限制计划最多领先车 1 m。触发时不仅把 `s_sched` 拉回来，还要**把 `schedule_t0` 回滚到对应时刻**。否则时钟会持续累积看不见的债务，车一追上参考就突然向前跳。
- **在线观测**：`/controller/schedule_lag_s`（正 = 落后，负 = 超前）。

A/B 对比：

```bash
roslaunch hofa_mpc_ros simulation.launch controller:=pid progress_mode:=projection
```

### 5.3 其他状态管理

- **`route_initial_speed` 每条路线只钉一次**。teacher 以 5 Hz 重发同一条路线，如果每次重发都重新读当前速度，剖面（以及计划时钟）会跟着测量漂移。更早的版本干脆不约束起点，结果计划在 `s = 0` 处就要求 0.2 m/s 而车还停着 —— 计划一出生就领先约 0.83 s / 0.17 m，控制器整段跑程都在追这个偏移。
- **`s_backtrack_tolerance_m = 0.5`**：允许扰动后有界回退，但阻止投影跳到更早的路线分支。
- **窗口锚点永不后退**：窗口可以随路线接近终点而缩短，但 `s_start` 绝不后移。后移会在车离目标最近的那一刻把参考放到车**后面**，两种控制器都会急刹并和它死锁。退化成 `s_start == s_end` 的零长窗口才是正确的"在终点保持"参考。

### 5.4 双窗口输出

| 输出 | 参数 | 话题 | 消费者 |
| --- | --- | --- | --- |
| 空间均匀窗口 | `n_resample: 20`，`lookahead_distance_m: 0.5` | `/controller/reference_path` | PID、可视化 |
| 时间均匀窗口 | `mpc_horizon_points: 20`，`mpc_dt_s: 0.1` | `/controller/reference_trajectory_window` | MPC（第 i 点对应 `t + i·dt`） |

时间窗口里的导数：加速度由前后点差分得到（端点用单侧差分），`dψ = v·κ`，`ddψ` 同样由前后点差分。

**`mpc_dt_s` 必须等于 `1 / mpc.control_rate_hz`** —— 控制器用自己的预测步长索引这个窗口，错配不会报任何错，只会让它跟踪错误时刻的点，看起来像调参问题而不是配置问题。`hofa_mpc_controller_node._check_reference_dt()` 在启动时校验并告警，同时也检查 `horizon` 是否超过窗口点数（超过的部分会用末点填充，不提供真实预览）。

---

## 6. ROS 控制节点 (`hofa_mpc_controller_node.py`)

### 6.1 坐标系转换

**这是本节点最大的复杂度来源**：Stonefish 用 NED/FRD，MPC 内部用 ENU/FLU。

| 位置 | 转换 |
| --- | --- |
| 里程计输入 | `x_enu = y_ned`，`y_enu = x_ned`，`ψ_enu = π/2 − ψ_ned`；机体速度 `u` 不变，`v` 和 `r` 取反 |
| 参考输入 | 同上，且**位置 / 速度 / 加速度三阶全部转换** |
| 力矩输出 | FLU → FRD：`Fy` 和 `Nz` 取反，`Fx` 不变 |
| 误差话题 | 转回 NED/FRD 再发布，以便和 `planar_pid_tracker` 在同一张 rqt_plot 上叠加 |

> **陷阱**：Stonefish 的 `Odometry::InternalUpdate` 在发布前已经应用了传感器基的逆变换，速度**已经在机体系**。不能当成世界系速度再旋转一次。

### 6.2 每周期流程

```
1. safety.check_state()            —— NaN / 超时 / 边界 / 超速 / 超转
2. safety.get_override_command(pre_solve=True)
3. 取参考窗口（不足 horizon 则用末点补齐，多余则截断）
4. Layer 1: predict_nominal_states → 逐步 compute_for_step → 冻结 Np 个 bounds
5. Layer 2: mpc.solve(state, refs, bounds)
6. hofa_inverse(state, a_c) → τ_enu
7. generalized_force_scale 逐轴缩放（正负向可分别设置，用于参数失配实验）
8. safety.validate_wrench() 硬限幅（力 ±100 N，力矩 ±50 N·m）
9. FLU → FRD，发布 WrenchStamped
```

### 6.3 发布话题

| 话题 | 类型 | 说明 |
| --- | --- | --- |
| `/controller/generalized_force` | `WrenchStamped` | 机体 NED/FRD 广义力 |
| `~status` | `ControllerStatus` | 状态、求解成功/迭代数/目标值、分层耗时、位置/偏航误差 |
| `~virtual_accel_cmd` | `AccelStamped` | MPC 输出的虚拟加速度 |
| `~predicted_path` | `Path` | 预测轨迹（转回 NED 可视化） |
| `/aquaflow/tracking_error/{x_body_m, y_body_m, yaw_rad, xy_norm_m}` | `Float64` | 与 PID 控制器话题名、单位、符号约定完全一致 |

误差标量**只在跟踪路径上发布**。早退分支不发零值，否则"控制器未使能"会在图上读成"完美跟踪"。

### 6.4 安全监督 (`safety.py`)

状态机：`DISABLED / WAITING_FOR_STATE / WAITING_FOR_REFERENCE / ACTIVE / DEGRADED / FAULT`

检查项（`config/safety.yaml`）：

| 检查 | 阈值 | 触发状态 |
| --- | --- | --- |
| NaN / Inf | — | FAULT |
| 状态超时 | 0.20 s | WAITING_FOR_STATE |
| 参考超时 | 0.30 s | WAITING_FOR_REFERENCE |
| 位置边界 | `[14.0, 8.0]` m | FAULT |
| 世界速度 | 1.0 m/s | FAULT |
| 偏航角速率 | 1.2 rad/s | FAULT |
| 连续求解失败 | 3 次 | FAULT |

> **`pre_solve` 的作用**：DEGRADED 分支在 `pre_solve=True` 时返回 `None`，**让位给求解器**。只有完成一次求解才会调 `on_solver_result()`，如果在求解前就抢先接管，`consecutive_failures` 会永远卡在 > 0 —— DEGRADED 变成既不恢复也不升级到 FAULT 的吸收态。再次失败会走调用方的失败路径进入 `pre_solve=False` 分支，保持上一次有效指令。

求解失败时会 `logwarn_throttle` 打印 scipy status、message、目标值、迭代数和连续失败计数 —— 静默的求解失败在操作员能看到的每一个话题上都和健康运行无法区分。

---

## 7. 辅助模块

### 7.1 基线控制器

- **`baseline_pd.py`** — computed-force PD，和 MPC 共用同一个 HOFA 逆变换与模型，用于公平对比。
- **`simulation.launch` 的 `controller:=pid`** — 实际走的是 `aquaflow_stonefish/planar_pid_tracker.py`（双闭环 PID：外环位置 P → 速度指令，内环速度 PI → 力），消费**同一个时间参考窗口**（`lookahead_time_s / mpc_dt_s` 选窗口索引），并加载**同一份车辆参数**做内环前馈。

### 7.2 轨迹发生器 (`trajectory.py` + `trajectory_server_node.py`)

离线测试轨迹：hover / line / circle / figure-eight / ellipse / S / rounded-rectangle。带五次样条（quintic spline）起步平滑，`yaw_mode` 可选 tangent 等。配置在 `config/trajectories.yaml`。

### 7.3 参数辨识 (`fit_drag_coefficients.py`)

**必须分两步，顺序很重要**：

1. `--mode drag` — 恒定推力，等待终速。稳态下 `F = D(u)`，惯性根本不出现，所以阻尼系数与质量无关。
2. `--mode added-mass` — 阶跃响应，第 1 步得到的阻尼**固定不动**，只剩惯性一个未知量。

一次性在同一段瞬态上拟合全部三个参数，正是当初拟合出**负横漂阻尼系数**的原因。

仿真参数现状（见 `vehicle_sim.yaml` 的警告）：只有 `drag_linear[0]` 来自稳态扫描，横漂和偏航的值是在错误质量下用瞬态数据拟合的，**不可信**，需要按两阶段重跑。

### 7.4 实验脚本 (`run_mpc_experiment.py`)

一键跑完整实验：启动仿真 → 运行 → 保存产物到 `results/mpc/run_YYYYMMDD_HHMMSS/`。

产物包括 `odometry.csv`、`controller_status.csv`、`generalized_force.csv`、`pwm.csv`、`mpc_reference_window.csv`、`evaluation_reference.csv`、`global_reference_path.csv`、`metrics.json`、`summary.txt`、`runtime_config.json`、`topics.bag`，以及四张图（`tracking_xy` / `tracking_errors` / `speed_tracking` / `control_wrench`）。

- **`runtime_config.json`** 记录本次运行从参数服务器实际读到的参数、关键源码和 YAML 的 SHA256、git commit、未提交改动、主机与 Python 信息。用来确认实验真的加载了当前代码和参数。
- **`--rescore <dir>`** 可对已有实验用新口径**离线重算**，不需要仿真器或 ROS master。在线离线共用同一个 `compute_metrics`，口径保证一致。

### 7.5 指标口径 (`metrics.py`)

指标严格分成**两个互不混淆的维度**：

| 维度 | 索引方式 | 指标 | 回答的问题 |
| --- | --- | --- | --- |
| 几何 | 按位置 | `cross_track_error_m`、`geometric_path_error_m`、`yaw_error_deg`、`speed_error_mps` | 在这个位置上，姿态和速度对不对 |
| 时序 | 按计划 | `progress_error_m`、`schedule_lag_s`、`time_lag_at_finish_s`、`completion_ratio` | 快了还是慢了 |

两个必读的坑：

1. **不要用 `mpc_reference_window.csv` 的 `index=0` 当评测参考**。那个点由 `s_start = s_proj` 生成，即车自身在路径上的投影 —— `planned_duration_s` 会恒等于本次运行时长、沿程进度误差恒为 0，时间维度上测不出任何东西。该 CSV 只作诊断用途。评测参考应由 `build_arc_profile` 用全局路径 + `/reference_processor` 的实际限幅参数**离线重算**。
2. **`speed_limit_violation_ratio` 不能单独看**。剖面的巡航速度**就等于** `max_speed`，所以零容差下任何居中的跟踪波动都会给出约 50% 的违规率 —— 这个比例分不出"贴着上限巡航"和"真超速"。必须结合 `mean_speed_mps`（均值是否高于上限）和 `speed_excess_mps`（超出幅度）一起判断。实际阈值是 `max_speed` 上浮 5%（`SPEED_LIMIT_TOLERANCE`），记录在 `speed_limit_threshold_mps`。

另有 `final_distance_to_goal_m`（比 `finish_reason` 更连续的量）和 `turn_yaw_error_deg` / `turn_yaw_rate_error_radps`（单独评估转弯阶段）。

### 7.6 测试

`test/` 下 9 个 pytest 文件：`test_model` / `test_hofa` / `test_mpc` / `test_constraints` / `test_allocator` / `test_safety` / `test_coordinates` / `test_trajectory` / `test_arc_profile`。

```bash
cd ~/catkin_ws/src/hofa_mpc_ros && python -m pytest test/
```

---

## 8. 配置文件索引

| 文件 | 加载位置 | 内容 |
| --- | --- | --- |
| `config/vehicle_sim.yaml` | 全局 + 控制器私有 | Stonefish 平台的惯性/阻尼/推进器构型 |
| `config/vehicle_real.yaml` | 同上（`platform:=real`） | 实物平台，多数为待辨识占位值 |
| `config/mpc.yaml` | 控制器私有 | 速率、horizon、权重、safe_box_scale、求解器、失败策略 |
| `config/safety.yaml` | 控制器私有 | 超时、边界、速度/转速限、失败计数 |
| `config/reference_processor.yaml` | `reference_processor` 私有 | 前视距离、重采样点数、速度/加速度限、锚点模式 |
| `config/simulator_interface.yaml` | 全局 | 话题名、坐标约定 |
| `config/trajectories.yaml` | `trajectory_server` | 离线测试轨迹定义 |

**跨文件的隐式耦合（改动时必须成对修改）**：

- `mpc.control_rate_hz` ↔ `reference_processor.mpc_dt_s`（必须互为倒数）
- `mpc.horizon` ≤ `reference_processor.mpc_horizon_points`

---

## 9. 快速上手

```bash
cd ~/catkin_ws && catkin_make && source devel/setup.bash

# 一键实验（MPC）
rosrun hofa_mpc_ros run_mpc_experiment.py --scene empty_pool

# 双环 PID 对照组
rosrun hofa_mpc_ros run_mpc_experiment.py --controller pid --scene empty_pool \
  --goal-x 4.0 --goal-y 0.0 --timeout 60

# 手动启动
roslaunch hofa_mpc_ros simulation.launch controller:=hofa_mpc platform:=sim

# 离线重算已有实验的指标
rosrun hofa_mpc_ros run_mpc_experiment.py --rescore results/pid/run_20260910_122124
```

更多运行参数见 [README.md](README.md)。

---

## 一句话概括

> 弧长重参数化 + 绝对计划时钟生成时间索引参考窗口 → HOFA 反馈线性化把非线性船体变成世界系双积分器 → Layer 1 用 LP 验证的内接盒把推进器限幅折算成虚拟加速度约束 → Layer 2 在盒约束下用解析梯度 L-BFGS-B 解 20 步线性误差 MPC → HOFA 逆变换补回科氏/阻尼/离心项得到机体广义力 → 加权伪逆分配到 4 个推进器。
