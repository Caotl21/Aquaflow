# Stonefish 中的 BricsBot 机器人模型与仿真环境

整理日期：2026-10-12。本文依据当前工作区场景、启动文件、动力学配置和推进器分配器整理，面向项目汇报；不涉及 hofa_policy 算法本身。

## 1. 总体仿真链路

Stonefish 1.6 + ROS Noetic 构成可重复的水下机器人仿真平台。BricsBot 在物理上是六推进器水下机器人；当前控制任务主要验证水平面三个自由度 surge、sway、yaw，垂向由 T5/T6 和外部深度控制链处理。

Stonefish 物理积分频率为 100 Hz，HOFA-MPC 控制频率为 10 Hz。闭环链路为：

Stonefish 水池与机器人动力学 → Odometry/IMU/Camera → 参考处理器 → HOFA-MPC 或 PID → WrenchStamped 广义力 → 六推进器分配器 → PWM → Stonefish。

hofa_mpc_ros/launch/simulation.launch 默认机器人为 bricsbot，默认 scene 为 random_pillars；aquaflow_pool.launch 负责把场景名映射到具体 .scn 文件。

## 2. 坐标系和 ROS 接口

Stonefish 场景使用 NED/FRD：世界 x 向北、y 向东、z 向下；机体 x 向前、y 向右、z 向下。接口配置为 world_ned 和 base_link。HOFA-MPC 内部使用 ENU/FLU，控制节点进行

$$
x_E=y_N,\qquad y_E=x_N,\qquad \psi_E=\operatorname{wrap}(\pi/2-\psi_N),
$$

$$
u_{FLU}=[u_{FRD},-v_{FRD},-r_{FRD}]^T,
\qquad \tau_{FRD}=\operatorname{diag}(1,-1,-1)\tau_{FLU}.
$$

Odometry 话题为 /bricsbot/odometry；控制器输入的广义力为 /controller/generalized_force，类型为 WrenchStamped；分配器输出 /bricsbot/setpoint/pwm 六路归一化命令。Stonefish 的 twist 按机体系速度处理，不能在控制器中当作世界系速度再次旋转。

## 3. BricsBot compound 机器人模型

### 3.1 外壳、质量块和浮力块

base_link 是 fixed=false、self_collisions=false 的 compound submerged 刚体。模型由三个物理部分组成：

1. 外部物理代理壳体：bricsbot_stonefish_physical_proxy_5000_no_top_bump_pre_refined_no_normals.obj，Fiberglass，厚度 0.004 m，显式质量 0.50 kg，不单独产生浮力。
2. 内部质量块：尺寸 0.374 x 0.358 x 0.058 m，Steel，质量 7.44 kg，NED z 方向偏移 +0.030 m，形成较低质心。
3. 内部浮力块：相同尺寸，Neutral，buoyant=true，质量字段 0.01 kg，NED z 方向偏移 -0.030 m。注释给出的排水体积是 7.775 L，在 1031 kg/m3 水体中提供约 0.8% 正浮力余量。

显式质量合计为 0.50 + 7.44 + 0.01 = 7.95 kg。质量块和浮力块位于机体原点两侧，形成静态恢复趋势。内部盒体是工程近似，不是完整 CAD 内部结构重建。

### 3.2 视觉模型与物理模型分离

视觉使用 bricsbot_stonefish_baked.obj 和 bricsbot/bricsbot.png；碰撞与水动力使用单独 physical proxy。物理和视觉网格均绕 x 轴旋转 -pi/2 以匹配 Stonefish NED，视觉再平移约 (-0.0008, 0, -0.0020) m 做包围盒居中。

convert_bricsbot_mesh.py 支持 OBJ/DAE 导入、比例缩放和 Decimate。这样可以独立替换碰撞代理、质量与浮力参数，而不必重新导出视觉 CAD。

## 4. 六推进器模型

四个水平推进器和两个垂向推进器的安装位置如下：

| 推进器 | 位置（m） | 作用 |
|---|---|---|
| T1 | (+0.12755, -0.12755, 0) | 水平 45°，surge/sway/yaw |
| T2 | (+0.12755, +0.12755, 0) | 水平 -45°，surge/sway/yaw |
| T3 | (-0.12755, -0.12755, 0) | 水平 135°，surge/sway/yaw |
| T4 | (-0.12755, +0.12755, 0) | 水平 -135°，surge/sway/yaw |
| T5 | (0, -0.1235, 0) | 竖直向上，depth |
| T6 | (0, +0.1235, 0) | 竖直向下，depth |

T1/T2 与 T3/T4 使用相反 handedness 的 propeller 网格；T2/T4 通过 180° 安装方向匹配力方向。T5/T6 通过绕 y 轴的相反 90° 安装产生相反竖直推力。

场景中的每个推进器使用 normalized_setpoint=true、inverted_setpoint=false、right=true、zero_order rotor dynamics，最大输入约 376.991118。zero_order 表示 Stonefish 当前没有显式一阶电机时间常数状态。

## 5. 推力曲线与分配

场景为每个推进器配置 39 点线性插值表，输入范围约 [-376.99, 376.99]，输出正反向不对称：

$$F_+^{max}=67.665885\ \mathrm{N},\qquad F_-^{max}=58.839900\ \mathrm{N}.$$

零附近存在死区，小幅 PWM 可能对应 0 N，因此不能用 F/Fmax 的单一线性比例替代。

brics6_thruster_allocator.py 接收

$$\tau_4=[F_x,F_y,N_z,F_z]^T$$

并建立 A in R^(4x6)，第 j 列为

$$A_{:,j}=[d_{x,j},d_{y,j},r_{x,j}d_{y,j}-r_{y,j}d_{x,j},d_{z,j}]^T.$$

分配公式为最小范数右伪逆：

$$f_6=A^T(AA^T)^{-1}\tau_4.$$

随后按正、反向曲线分段线性反查 PWM；绝对推力不超过 0.05 N 时置零，曲线端点饱和到命令 ±1。广义力超过 0.25 s 未更新或节点关闭时，输出六路零命令。

这是无推力上下界的伪逆，不是精确有界二次规划；也没有显式推进器变化率约束。因此 HOFA-MPC Layer 1 的内接盒证明“存在可行推力”，不能直接推出伪逆选出的那组推力永不饱和。

## 6. Stonefish 水池和水体

当前场景统一使用：水密度 1031 kg/m3，Jerlov 0.15；波高 0；颗粒系统关闭；均匀流速为 (0,0,0) m/s；太阳方位角 45°、高度角 60°。材料包括 Neutral 1000、Rock 3000、Fiberglass 1500、Steel 7850 kg/m3；障碍场景还定义 Aluminium 2710 kg/m3。

基础水池为 14 m x 8 m，水深 4 m。水面 z=0；池底板厚 0.2 m，中心 z=4.1 m；四面墙厚 0.2 m，中心位于 x=±7.1 m、y=±4.1 m，高度 4 m。普通多障碍场景中 BricsBot 从 (-5,0,1) m、零 yaw 开始；独立 bricsbot_pool.scn 推力测试从 (0,0,1) m 开始。

## 7. 实验场景

| 场景 | 用途 | 设置 |
|---|---|---|
| empty_pool | 无障碍基线 | 只有池底和四壁 |
| empty_pool_control | 无相机 headless 控制采集 | 物理与空池一致，省略 camera_front |
| single_obstacle | 单障碍绕行 | 中心圆柱，半径 0.45 m，高 4 m，中心 (0,0,2) |
| random_pillars | 多障碍、多尺度环境 | 固定坐标的伪随机圆柱，直径约 0.015–0.60 m，贯穿全水深 |

random_pillars 不是运行时随机生成，而是固定写入场景文件，因此同一版本可重复。aquaflow_pool.launch 对未知 scene 名称不会静默回退。hofa_mpc_ros 默认 random_pillars；做纯控制基线时应显式使用 empty_pool。

## 8. 传感器和 TF

| 传感器 | 频率 | 配置 |
|---|---:|---|
| Odometry | 100 Hz | /bricsbot/odometry |
| Pressure | 5 Hz | /bricsbot/pressure |
| Front camera | 30 Hz | 640x480，水平 FOV 80°，安装约 (0.20,0,-0.03) m |
| IMU | 20 Hz | 角度噪声约 1.745e-6，角速度噪声约 1.745e-5 |

相机最小/最大渲染距离为 0.05/100 m，frame_id 为 bricsbot/camera_front。普通场景由 launch 补充 bricsbot/base_link 到 bricsbot/camera_front 的静态 TF；无相机控制场景不启动该 TF。

## 9. Stonefish 被控对象和 HOFA 标称模型的关系

必须区分仿真器实际积分的模型和控制器使用的标称模型。当前 vehicle_sim.yaml 为：

$$M_{dry}=diag(7.95,7.95,0.174887),$$
$$M_A=diag(26.254017,26.254017,0.067653),$$
$$M=diag(34.204017,34.204017,0.242540).$$

阻尼为

$$d_l=[35.488892,35.656137,1.163915]^T,
\qquad d_q=[17.853671,17.584671,0.021297]^T.$$

Stonefish physical proxy 的平移附加质量来自 MVAE/几何估计，当前控制器采用 26.254017 kg 的平移平均值；偏航 0.067653 kg m2 是有效辨识项。这里的附加质量是场景级等效参数，不能直接解释成实体机器人水池测量值，也不能和 vehicle_real.yaml 互换。

场景中的 compound 质量块是 7.95 kg；vehicle_sim.yaml 中 added_mass 是控制器模型的附加惯性，不是在场景里再增加一个可见刚体。控制器当前 coriolis_enabled=false。

## 10. 建模假设与限制

1. 主要验证水平 3-DOF 跟踪；垂向推进器存在，但深度控制与水平控制分层。
2. 默认环境为无波、无流、无颗粒，不能代表真实流场或波浪扰动。
3. Stonefish 推进器使用 zero_order；配置中的 time_constant_s=0.10 s 不会自动变成场景中的一阶执行器状态。
4. 外部六推分配使用无约束伪逆，饱和后不重新优化；实验应记录分配饱和和实际推力残差。
5. physical proxy、内部质量盒、浮力盒和附加质量是工程近似，不是 CAD 级 CFD。
6. simulation_rate=100 是物理步频，控制器 10 Hz 是闭环命令频率，不能混写。
7. empty_pool_control 适合 headless 控制采集，但不能用于视觉性能评价。
8. BlueROV2 场景仅作对照，不能把其模型或推进器参数混入 BricsBot 结论。

## 11. 汇报版总结

我们在 Stonefish 中建立了一个可重复的 BricsBot 水下仿真平台。机器人采用物理代理壳体、显式质量块和浮力块组成的 compound 模型，匹配了总质量、质心/浮心分离和六推进器安装几何；推进器使用实测非对称正反向推力曲线，并通过六推伪逆分配器接收 HOFA-MPC 输出的水平广义力。环境采用 14 m x 8 m x 4 m 的无流静水池，提供空池、单障碍和固定多尺度障碍三类可复现实验场景；传感器链包含 100 Hz 里程计、30 Hz 相机、20 Hz IMU 和 5 Hz 压力。

这套模型的优势是参数可追溯、场景可复现、接口与真实六推进器链一致；主要边界是水动力代理和附加质量属于场景级近似，环境暂不包含流、波和颗粒，推进器执行器采用零阶响应，外部伪逆分配没有显式有界优化。汇报中应把 Stonefish 被控对象、HOFA-MPC 标称模型和真实机器人参数三者分开，避免把等效附加质量或仿真结果表述成实体机器人实测结论。

## 12. 代码索引

- aquaflow_stonefish/scenarios/bricsbot_reference.scn
- aquaflow_stonefish/scenarios/bricsbot_control_reference.scn
- aquaflow_stonefish/scenarios/aquaflow_empty_pool.scn
- aquaflow_stonefish/scenarios/aquaflow_single_obstacle.scn
- aquaflow_stonefish/scenarios/aquaflow_random_pillars.scn
- aquaflow_stonefish/launch/aquaflow_pool.launch
- hofa_mpc_ros/launch/simulation.launch
- aquaflow_stonefish/scripts/brics6_thruster_allocator.py
- hofa_mpc_ros/config/vehicle_sim.yaml
- hofa_mpc_ros/config/simulator_interface.yaml
