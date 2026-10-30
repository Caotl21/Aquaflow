# Stonefish Bearing-angle 粒子滤波避障实验开发文档

## 1. 文档目的

本文档定义将 `bearing_angle_avoidance_demo` 迁移到 Stonefish + BricsBot 的开发路线。目标是在不向在线规划器提供真实障碍物位置、真实距离或 privileged map 的条件下，使用 Stonefish 前视相机和里程计完成：

```text
黑色柱体图像 -> bearing-angle 观测 -> 粒子滤波估计 -> 局部避障规划
-> 局部轨迹跟踪 -> BricsBot/Stonefish 动力学执行
```

第一版只验证水平面二维避障。柱体固定、贯穿水深，机器人保持目标深度；真实碰撞和真实地图只允许出现在离线评价中。

## 2. 当前代码与 Stonefish 基础

### 2.1 粒子滤波避障基线

当前演示位于：

```text
src/bearing_sim/bearing_angle_avoidance_demo/bearing_avoidance/
```

主要文件：

- `avoidance_demo.py`：场景、角度传感器、目标管理、规划器、闭环执行和绘图。
- `bearing_angle_pf.py`：静态圆柱 `[cx, cy, L]` 的 bearing-angle 粒子滤波器。
- `test_core.py`：线段碰撞、遮挡、部分轮廓、ID 保持和规划器门控检查。
- `evaluate_seeds.py`：固定参数的批量实验。

当前算法使用完整观测 `[bearing, angular_width]`，并假定跨帧 ID 已关联。规划器只读取位姿、目标、后验地图样本和部分轮廓角区间，不读取真实障碍物数组。

### 2.2 已有 Stonefish 资产

- `src/aquaflow_stonefish/scenarios/aquaflow_random_pillars.scn` 已定义黑色外观 `pillar_black` 的静态圆柱。场景柱体贯穿水深，直径约为 `0.015--0.60 m`，布局固定，便于复现。
- `src/aquaflow_stonefish/scenarios/bricsbot_reference.scn` 已定义 BricsBot 前视相机：30 Hz、640x480、水平 FOV 80°，话题前缀为 `/bricsbot/camera_front`。
- 相机输出为 `/bricsbot/camera_front/image_color` 和 `/bricsbot/camera_front/camera_info`。
- 里程计为 `/bricsbot/odometry`，低层控制链已经可以通过 PID 或 HOFA-MPC 驱动六推进器。
- `src/aquaflow_stonefish/launch/tracking.launch` 可以接收外部路径和速度参考，适合作为第一版低层执行链。

## 3. 实验边界与不变量

### 3.1 在线规划器允许读取的数据

```text
- 相机图像或分割图像
- camera_info
- BricsBot 里程计位姿和速度
- 目标点
- 自己维护的粒子滤波后验
```

### 3.2 在线规划器禁止读取的数据

```text
- Stonefish 场景中的柱体数组
- random_pillars.yaml 中的障碍物坐标
- 深度相机或任何直接距离测量
- 真实碰撞结果
- privileged teacher 的避障路径
```

真实柱体位置可以保存到 `episode.npz` 或评价日志中，但只能由离线评价器读取。

### 3.3 第一版物理假设

- 柱体静止、竖直并贯穿水深，水平截面近似圆形。
- 机器人只评价 `x-y` 平面运动，深度由现有控制链保持在约 `z=1.0 m`。
- 初期使用无流、无波、无颗粒的 Stonefish 环境，稳定后再加入流场和感知扰动。
- 粒子滤波仍使用直径 `L`，Stonefish XML 和地图文件中的 `radius` 必须转换为 `L=2*radius`。

## 4. 目标系统架构

```text
/bricsbot/camera_front/image_color       /bricsbot/camera_front/camera_info
                    |                                  |
                    +---------- observation_adapter --+
                                     |
                     AngularObservation
                       (id, beta, theta, partial sectors)
                                     |
                              Tracker / PF
                                     |
                              sampled maps
                                     |
                                  plan()
                                     |
             local Path + velocity reference in world_ned
                                     |
                  planar PID or HOFA-MPC controller
                                     |
             generalized force -> brics6_thruster_allocator
                                     |
                                  Stonefish
```

第一版建议使用 PID 执行局部轨迹，减少动力学调参对感知和规划验证的干扰；粒子滤波规划稳定后再切换 HOFA-MPC。

## 5. 观测适配器设计

### 5.1 分割相机验证模式

Stonefish 支持 `type="segmentation"` 的分割相机。ROS 输出包括 16 位物体 ID 图像、相机信息和可视化图像。该模式不提供距离，但提供理想的对象 ID，因此适合先验证几何投影、PF 和规划器；实验报告必须标注这是“理想关联”结果。

建议在独立的实验场景中增加第二个相机，不覆盖现有 RGB 相机：

```xml
<sensor name="bearing_segmentation" rate="30.0" type="segmentation">
  <specs resolution_x="640" resolution_y="480" horizontal_fov="140.0"/>
  <origin xyz="0.20 0.0 0.0" rpy="1.5707963268 0.0 1.5707963268"/>
  <link name="base_link"/>
  <ros_publisher topic="/bricsbot/bearing_segmentation"/>
</sensor>
```

适配器对每个柱体 ID 做水平像素统计：

```text
u_left  = mask 的最小列
u_right = mask 的最大列
u_center = 0.5 * (u_left + u_right)

beta  = atan((u_center - cx) / fx)
theta = atan((u_right - cx) / fx)
        - atan((u_left - cx) / fx)
```

`cx` 和 `fx` 必须来自 `camera_info`。完整轮廓产生 `[beta, theta]`；被其他柱体遮挡时，只保留实际可见角区间，不能把残缺轮廓当成完整角宽。

### 5.2 RGB 黑柱模式

第二阶段使用现有 `/bricsbot/camera_front/image_color`：

1. 转换到灰度或 HSV 空间。
2. 根据黑柱外观建立低亮度候选 mask。
3. 去除过小区域、池壁、底面和机器人自身区域。
4. 对连通域提取水平左右边界。
5. 使用上一帧 bearing、角宽和里程计预测进行最近邻 ID 关联。
6. 在关联失败时保留滤波器后验，但不伪造完整观测。

RGB 模式必须加入漏检、误检、像素量化和遮挡测试。不能把分割相机 ID 直接作为 RGB 模式的真实 ID。

当前节点还支持 `sensor_mode:=truth_angle` 作为 privileged 几何对照。该模式直接用固定 P1 场景中的柱体真值计算 bearing 和 angular width，只把这两个角度量送入 PF，不把距离或柱体坐标送入 planner；它用于先排除图像投影、分割和关联问题，不能作为感知性能结果。`sensor_mode:=rgb` 仍只使用黑色像素、形状门控和基于 bearing 的短时最近邻关联；它不读取 Stonefish 分割 ID、深度或地图。默认 RGB 输入为 `/bricsbot/camera_front/image_color`，mask 可视化发布到 `/bearing_pf/rgb_mask`。

P1 的局部参考深度固定为 `target_depth_m:=1.0`。节点发布的每个局部路径点都使用这个深度，现有 PID 用深度误差保持机器人在水下 1 m；RGB mask 同时拒绝横向宽条带，避免把水面或池底当成竖直柱体。

注意：`enable:=false` 会关闭六推进器 allocator，因此不会执行任何定深。若要“只定深、不水平移动”，必须使用 `enable:=true motion_enabled:=false`；此时 PF 仍处理图像，但节点只发布当前位置的水平保持参考，PID/allocator 只负责保持 `z=1.0 m`。

如果“椭圆”只是透视图像中的轮廓，当前 `[bearing, angular_width]` 圆柱模型仍然可以使用；如果柱子的真实水平截面是椭圆，当前 PF 模型不够，需要把状态从 `[cx, cy, diameter]` 扩展为椭圆长短轴和朝向。

## 6. 坐标和时间约定

Stonefish 使用 `world_ned / base_link` 和 FRD 约定。粒子滤波器可以继续使用世界 `x-y` 平面，但必须明确：

- 里程计位置和目标使用 `world_ned`。
- 里程计 yaw 从姿态四元数提取，符号保持与当前 Stonefish 控制链一致。
- 相机相对机体的水平偏移为约 `(+0.20, 0.0) m`，PF 首次观测的 anchor 必须是相机位置，而不是机体中心。
- 相机正前方对应机体 `+x`，图像水平右侧对应机体 `+y`；bearing 的正负号需要用一帧已知柱体做单元测试确认。
- 图像时间戳、里程计时间戳和规划周期必须记录。禁止用最新里程计无条件配对旧图像。

推荐频率：相机 30 Hz，观测适配和 PF 10--20 Hz，局部规划 5--10 Hz，Stonefish 物理积分 100 Hz。

## 7. 局部规划和控制接入

当前 `plan()` 返回恒速二维速度，而 Stonefish 需要经过动力学控制。因此适配器应把一次规划结果展开为短时间局部参考：

```text
p_i = p_now + v * (i * dt_ref)
yaw_i = atan2(v_y, v_x)
v_i = v
```

第一版发布：

```text
/bearing_pf/local_path      nav_msgs/Path
/bearing_pf/local_velocity  geometry_msgs/TwistStamped
```

然后使用：

```bash
roslaunch aquaflow_stonefish tracking.launch \
  scene:=random_pillars \
  enable:=true \
  reference_topic:=/bearing_pf/local_path \
  velocity_reference_topic:=/bearing_pf/local_velocity
```

这条命令假定后续新增的 Stonefish 规划节点已经发布上述两个话题。规划节点不能同时启动 privileged teacher。

稳定后可接入 `hofa_mpc_ros/launch/simulation.launch`，让局部参考进入 `/controller/reference_trajectory_window`。MPC 版本需要额外检查参考窗口更新时间、局部重规划造成的路径跳变和推进器饱和。

## 8. 场景策略

### 8.1 第一版场景

不要直接把 41 根、直径跨越三个数量级的柱体作为首个验收场景。建议复制 `aquaflow_random_pillars.scn` 为实验场景，并保留 6--12 根直径 `0.15--0.60 m` 的黑色柱体：

- 至少 2--3 根柱体位于起点到目标的直线上。
- 柱体间留出明确可通行间隙。
- 保持柱体贯穿水深，使二维圆柱模型成立。
- 固定场景文件，先保证一次实验可重复。

当前 Stonefish 相机为 80°，而 demo 使用 140°。为了与二维基线直接比较，第一版建议实验相机使用 140°；随后将 80°作为受限视场消融实验。若坚持 80°，必须同步缩小 planner 的 heading 候选范围，不能继续假设 `+-75°` 都可执行。

### 8.2 后续场景扩展

按以下顺序增加难度：

1. 固定 3 根大柱体，验证闭环和碰撞评价。
2. 固定 6--12 根中等柱体，验证多目标 PF 和遮挡。
3. 生成多个确定性场景变体，验证跨布局泛化。
4. 加入小柱体、漏检、误检、流场和姿态扰动。

直径接近 `0.015 m` 的柱体在远距离下可能只有一两个像素，不能把“未检测到”解释为算法安全地绕开了该柱体。

## 9. 实施阶段

### P0：离线接口

- 从现有 demo 导入 `BearingAnglePF`、`Tracker` 和风险计算。
- 用人工生成的 mask 验证像素到 bearing/角宽的投影公式。
- 验证 NED/FRD yaw、相机偏移和直径/半径转换。
- 为观测适配器建立不依赖 ROS 的单元测试。

验收：固定 mask 输入下，bearing 和角宽误差小于设定像素量化误差；PF 权重有限且和为 1。

### P1：Stonefish 分割相机闭环

- 增加实验场景和分割相机。
- 订阅分割图、camera_info、odometry。
- 使用分割 ID 形成理想关联的 `AngularObservation`。
- 通过 PID 执行局部路径。
- 记录图像、观测、PF 估计、规划风险、里程计和推进器命令。

验收：至少一个固定场景完整到达，在线规划器不读取地图和深度，真实碰撞为零。

### P2：RGB 黑柱检测

- 使用 RGB 图像生成黑柱候选 mask。
- 增加连通域筛选、ID 关联、漏检和残缺轮廓处理。
- 保持 PF 与 planner 接口不变，只替换 observation adapter。

验收：在相同场景下对比 P1，报告检测召回、关联错误、估计误差和避障结果。

### P3：动力学与控制对比

- 对比 PID 和 HOFA-MPC。
- 记录实际速度、参考速度、广义力、PWM 和推进器饱和。
- 检查局部重规划对参考窗口和控制器稳定性的影响。

验收：控制器能跟踪局部路径，速度不持续超限，规划停顿不会造成旧参考继续推进。

### P4：鲁棒性和泛化

- 多个确定性柱体布局。
- 相机噪声、漏检、遮挡、图像延迟。
- 小幅流场、姿态变化和不同安全余量。
- 与直线基线、已知地图规划器和分割相机版本比较。

## 10. 日志和评价产物

每个运行目录建议至少包含：

```text
summary.json                 状态、步数、路径长度、最小净空
observations.csv             每帧检测、bearing、角宽、关联 ID
particle_estimates.csv       每个目标的后验均值和不确定性
planner_history.csv          候选风险、选中速度、停止原因
odometry.csv                 Stonefish 实际位姿和速度
generalized_force.csv        控制器广义力
pwm.csv                      六推进器命令
episode.npz                  离线评价所需轨迹和真实场景
topics.bag                   必要 ROS 话题录包
```

在线规划器不读取 `episode.npz` 中的真实柱体；评价脚本可以使用它计算：

- 是否到达目标。
- 是否发生真实几何碰撞。
- 执行轨迹最小表面净空。
- 路径长度和模拟时间。
- 每个已关联柱体的中心/直径误差。
- 后验经验风险与真实碰撞结果的对应关系。

## 11. 主要风险与处理方式

| 风险 | 处理方式 |
| --- | --- |
| 80° FOV 导致避障方向不足 | 首版使用 140°；80°作为单独消融 |
| 黑色阈值受水下光照影响 | 先用分割相机验证，再做 RGB mask 和参数扫描 |
| 分割 ID 造成关联性能虚高 | 报告为 oracle association；P2 使用 RGB 关联 |
| 小柱体低于像素分辨率 | 首版限制直径，后续提高分辨率或缩短范围 |
| 相机和机体位置不一致 | 用 camera TF 计算 PF anchor |
| demo 的瞬时速度不符合 Stonefish 动力学 | 发布局部轨迹，交给 PID/MPC 跟踪 |
| Stonefish NED 与其他模块坐标混用 | 所有消息明确 frame_id，并做坐标单元测试 |
| 规划器误读真实地图 | 节点接口只接收图像、camera_info、odom、goal 和后验 |
| 固定场景结果被误读为成功率 | 使用多个确定性布局，报告场景数量和失败原因 |

## 12. 当前结论

该实验在现有工作区中是可实现的。首个可交付版本应选择“固定黑柱场景 + 分割相机理想关联 + PID 局部轨迹跟踪”，先验证粒子滤波和避障闭环；之后只替换观测适配器为 RGB 黑柱检测，形成更接近真实视觉条件的版本。Stonefish 的真实地图、深度信息和碰撞结果只能用于离线评价，不能进入在线 planner。

## 13. 验证命令

### 13.1 P0：纯 Python 检查

在安装了 `numpy` 的 Python 环境中执行：

```bash
cd /home/bricsbot/catkin_ws/src/bearing_sim/bearing_angle_avoidance_demo/bearing_avoidance
python3 test_stonefish_observation.py
python3 test_core.py
python3 test_rgb_association.py
python3 test_stonefish_recovery.py   # 用桩模块加载节点，不需要 ROS
```

预期输出分别包含：

```text
PASS: camera projection, scene-label filtering, partial sectors, ROS 16UC1 decoding
PASS: geometry, continuous collision, occlusion, truncation, ID retention, planner gating
```

### 13.2 编译并检查新 ROS 节点

在装有 ROS Noetic 和 Stonefish 的机器上执行：

```bash
cd /home/bricsbot/catkin_ws
catkin_make --pkg aquaflow_stonefish
source /home/bricsbot/catkin_ws/scripts/setup_stonefish_ros.sh
```

### 13.3 P1：只启动仿真和观测规划，不驱动推进器

先启动 ROS master：

```bash
roscore
```

在另一个终端执行：

```bash
source /home/bricsbot/catkin_ws/scripts/setup_stonefish_ros.sh
mkdir -p /tmp/bearing_pf_p1_observation
roslaunch aquaflow_stonefish bearing_pf_stonefish.launch \
  enable:=false \
  log_dir:=/tmp/bearing_pf_p1_observation
```

在另一个终端读取一帧分割图中的对象 ID：

```bash
source /home/bricsbot/catkin_ws/scripts/setup_stonefish_ros.sh
rosrun aquaflow_stonefish inspect_segmentation_labels.py
```

当前固定 P1 场景的柱体 ID 为 `[12,14,16,18,20,22,24,26]`。本帧没有出现的 `14`、`16` 是被前方柱体遮挡的目标，仍要保留在列表中。`2、6、8、10` 属于池底/池壁，不要加入列表。若修改场景对象顺序，需要重新检查 ID。

再开一个终端检查输入和输出：

```bash
source /home/bricsbot/catkin_ws/scripts/setup_stonefish_ros.sh
rostopic list | grep -E 'bearing_segmentation|bricsbot/odometry|bearing_pf/local'
rostopic hz /bricsbot/bearing_segmentation/image_raw
rostopic hz /bricsbot/odometry
rostopic echo -n 1 /bearing_pf/local_path
rostopic echo -n 1 /bearing_pf/local_velocity
cat /tmp/bearing_pf_p1_observation/observations.csv
```

这里 `enable:=false` 只关闭 PID 推力输出；分割图、PF、局部路径和观测日志仍然运行。确认相机话题有数据、`local_path` 不为空、日志中能看到柱体 label 后，再进入闭环。

### 13.4 P1：启动实际闭环

```bash
source /home/bricsbot/catkin_ws/scripts/setup_stonefish_ros.sh
mkdir -p /tmp/bearing_pf_p1_closed_loop
roslaunch aquaflow_stonefish bearing_pf_stonefish.launch \
  enable:=true \
  goal_x:=5.0 \
  goal_y:=0.0 \
  allowed_labels:="[12,14,16,18,20,22,24,26]" \
  log_dir:=/tmp/bearing_pf_p1_closed_loop
```

闭环运行期间观察：

```bash
rostopic echo /bearing_pf/status
rostopic echo /aquaflow/tracking_error/xy_norm_m
rostopic echo /bricsbot/setpoint/pwm
rostopic echo /bearing_pf/local_velocity
```

完成一次运行后按 `Ctrl-C` 停止，并保留 `/tmp/bearing_pf_p1_closed_loop/observations.csv`、终端日志和 Stonefish 录屏。该版本使用分割相机的理想 ID 关联，结果只能作为 P1 几何闭环验证，不能作为 RGB 检测性能结论。

### 13.5 运行限制

不要在 P1 命令中同时启动 `hofa_mpc_ros/launch/simulation.launch` 或 privileged teacher。P1 的在线规划器只应读取分割图、camera_info、里程计和目标；真实场景文件只用于 Stonefish 生成环境和离线评价。

### 13.6 RViz 调试与碰撞归因

启动时加 `rviz:=true` 会加载专用配置：

```bash
roslaunch aquaflow_stonefish bearing_pf_stonefish.launch \
  enable:=false \
  rviz:=true \
  allowed_labels:="[12,14,16,18,20,22,24,26]" \
  log_dir:=/tmp/bearing_pf_p1_debug
```

RViz 中的 `/bearing_pf/debug_markers` 各层含义：

- 绿色扇区：当前分割相机完整柱体观测；橙色扇区：残缺轮廓禁行区。
- 青色点和椭圆：PF 后验粒子和约 2-sigma 的位置不确定性。
- 橙色实心圆和轮廓：PF 后验均值的当前障碍物估计；橙色轨迹：该估计中心的历史移动轨迹。
- 红色实心圆和轮廓：planner 实际使用的膨胀圆，半径为 `diameter/2 + robot_radius + safety_margin`。
- 蓝色圆：机器人当前 footprint；绿色线：选中的局部线段；红色线：planner 停止。
- 文字中的 `risk`、`clear5`、`R`、`M` 分别是经验碰撞风险、后验 5% 净空、机器人半径和安全余量。

归因规则：

1. 红色膨胀圆已经与真实柱体重叠，或青色粒子明显偏离柱体：先判断估计/观测关联问题。
2. 红色膨胀圆没有覆盖真实柱体：膨胀半径不足或柱体直径估计偏小；先增大 `safety_margin_m`，不要先调 PID。
3. 红色膨胀圆覆盖真实柱体、`clear5` 为正，但实际轨迹撞击：planner 后验判断与执行器轨迹不一致，应检查 PID 跟踪、参考更新时间、相机/里程计时间偏差和实际 footprint。
4. 绿色轨迹直接穿过红色膨胀圆：规划器/风险门控有问题；检查 `risk_threshold`、粒子地图采样和 `clearance_5p_m`。

运行后新增的诊断文件为 `odometry.csv`、`particle_estimates.csv`、`planner_history.csv`。可用固定 P1 场景离线复核：

```bash
python3 /home/bricsbot/catkin_ws/scripts/evaluate_bearing_pf_stonefish_run.py \
  /tmp/bearing_pf_p1_debug
cat /tmp/bearing_pf_p1_debug/diagnosis.json
```

这个脚本只用于离线归因，在线 PF 节点不会读取真实场景几何。

RViz 真值层（仅显示）：`truth_obstacle`（真实半径，浅灰）、`truth_inflated`（加机器人半径与安全余量，深灰）、`truth_labels`（`T01` 起，按 yaml 顺序 = `BearingPillarNN`）。由 launch 参数 `show_truth` 控制，默认在 P1 场景或传了 `truth_pillars_file` 时打开。密集场景的 RGB 运行需加 `truth_pillars_file:=.../dense_s10X.yaml` 才会显示，该参数在 rgb 模式下不参与关联、PF 或规划。

### 13.7 RGB 黑柱模式验证

先停掉旧 launch，再使用真实 RGB 相机：

```bash
roslaunch aquaflow_stonefish bearing_pf_stonefish.launch \
  sensor_mode:=rgb \
  image_topic:=/bricsbot/camera_front/image_color \
  camera_info_topic:=/bricsbot/camera_front/camera_info \
  planner_fov_deg:=80.0 \
  target_depth_m:=1.0 \
  motion_enabled:=false \
  enable:=true \
  rviz:=true \
  log_dir:=/tmp/bearing_pf_rgb_debug
```

检查黑色 mask：

```bash
rostopic hz /bearing_pf/rgb_mask
rostopic echo -n 1 /bearing_pf/status
```

RViz 里的 `RGB Camera` 和 `/bearing_pf/rgb_mask` 可用于判断颜色阈值；若 mask 包含池底或机器人外壳，先调 `rgb_dark_max_value`、`rgb_dark_channel_spread`，暂时不要调 PF。

RGB 检测器按“地平线带”找柱：只统计 `cy±rgb_band_half_height_px` 行内暗像素比例 ≥ `rgb_band_fill` 的列，连续暗列就是一根柱的左右边界。柱体贯穿水深，在相机深度一定穿过地平线；水面和池底不会，因此不再需要宽度/长宽比门控。近处很宽的柱照常输出；贴图像左右边缘的柱输出为残缺轮廓禁行扇区。远处柱被水体染成蓝色（P1 场景 4–9 m 处通道差约 35–57，背景约 130），所以默认阈值为 `rgb_dark_max_value=80`、`rgb_dark_channel_spread=60`。

图像上重叠的柱会先连成一段暗列。单根柱在地平线带内的亮度和落地行几乎恒定，而远处柱更亮（水体衰减）、落地行更高，所以在亮度阶跃 ≥ `split_brightness=6` 或落地行阶跃 ≥ `split_bottom_px=12` 处切开；较近的一段保持完整，被挡住的较远一段输出为残缺轮廓。距离和亮度都接近、且都伸出画面底部的两根柱仍会合并。

RGB 关联（`rgb_association.py`）不再用上一帧像素 bearing 做最近邻：每条轨迹用自己的 PF 后验粒子从当前相机位姿投影出预测 bearing/角宽，门限为 `rgb_match_gate_min_deg + 3σ`，上限 `rgb_match_gate_max_deg`。预测已出画面的轨迹不参与匹配，避免刚绕过的柱子“认领”随后在相近像素位置出现的新柱（`rgb_fov80_1029_1256` 中 A 的轨迹吞掉了 B 的全部观测，导致 B 从未进入地图）。贴边的残缺分量不更新任何轨迹；角宽与已收敛轨迹预测相差超过 `rgb_width_tolerance`（相对）+3σ 的检测视为重叠/遮挡，只作为禁行扇区，不更新 PF。

`rgb_fov80_1029_1316` 卡死的原因和对应修改：

- 一帧 1 px 宽的误检（`rgb_6`）建立了后验极散（±2 m）的轨迹，其 10° 门限覆盖柱 C；C 的检测先匹配到它、宽度不符又被当成禁行扇区，C 从未建图，且每帧都带 ±8° 余量封住前方。现在宽度检查在配对时进行：宽度不符的配对直接放弃，只有 bearing σ ≤ 2° 的已收敛轨迹才能把“过宽”检测判为重叠扇区。
- 新轨迹要求角宽 ≥ `rgb_min_new_width_deg=0.6`，且离画面边缘 ≥ 1°（`rgb_8` 是贴边 0.25° 的 E，产生了 L=0.54 m 的重复轨迹）。
- 检测次数少于 `rgb_confirm_hits=3` 的暂定轨迹，在预测位于画面内的情况下累计 `rgb_tentative_timeout_s=2.0` 秒没再被看到就删除；预测在画面外的时间不计入（`rgb_fov80_1029_1528` 中 B 只在画面边缘被看到 2 次，出视场后被删，机器人经过 B 时地图里没有它）。
- `planner_history.csv` 新增 `partial_sectors_deg` 列（相对相机，`左:右;…`）。
- 规划器连续保持 `stuck_timeout_s=3` 秒后进入恢复：忽略视场/扇区限制、只按当前后验找低风险方向，原地转向该方向（`reason=recover_turn`）；仍没有就沿当前朝向以 `recovery_backoff_speed_mps=0.1` 后退（`recover_backoff`，相机保持朝前）。

```bash
cd src/bearing_sim/bearing_angle_avoidance_demo/bearing_avoidance
python3 test_rgb_association.py
```

单帧调阈值：

```bash
rosrun aquaflow_stonefish inspect_rgb_pillars.py _out_dir:=/home/bricsbot/catkin_ws/results/bearing_pf/rgb_inspect
```

### 13.8 Privileged 真值角度对照

先用真值角度模式隔离 PF、规划器和控制链；该模式不需要相机图像或 `camera_info`：

```bash
roslaunch aquaflow_stonefish bearing_pf_stonefish.launch \
  sensor_mode:=truth_angle \
  planner_fov_deg:=140.0 \
  enable:=true \
  rviz:=true \
  log_dir:=/tmp/bearing_pf_truth_angle
```

真值角度模式只发布每个可见柱体的 `[bearing, angular_width]`。距离仍由 PF 根据运动视差估计；如果改为直接传距离或柱体坐标，就不再是 bearing-angle 对照实验。

默认的真值模式无噪声、无遮挡，被前柱挡住的柱体也会被“看到”，因此明显比相机模式乐观。与 RGB 模式做公平对照时，使用相同 FOV 并打开遮挡和噪声：

```bash
roslaunch aquaflow_stonefish bearing_pf_stonefish.launch \
  sensor_mode:=truth_angle \
  planner_fov_deg:=80.0 \
  truth_occlusion:=true \
  truth_noise_beta_deg:=0.6 \
  truth_noise_theta_deg:=0.35 \
  enable:=true \
  rviz:=true \
  log_dir:=/home/bricsbot/catkin_ws/results/bearing_pf/truth_angle_fov80_occ_noise
```

- `truth_occlusion:=true`：按射线投射判断遮挡；完全被挡住的柱体不输出，部分可见或被 FOV 截断的柱体只输出可见角区间作为残缺轮廓禁行区。
- `truth_noise_*_deg`：高斯角度噪声，`0.6/0.35` 与 PF 似然中的 sigma 一致。

离线评价脚本同时支持场景 ID（分割/真值模式）和 RGB 的 `rgb_N` 轨迹。RGB 轨迹按最终后验均值匹配到最近的真实柱体，超过 `--match-gate`（默认 1.0 m）记为虚假轨迹；`diagnosis.json` 的 `association` 字段报告未跟踪柱体、一柱多轨迹（ID 碎片化）和虚假轨迹，`estimate_errors` 中的 `center_error_at_closest_pass_m` 是机器人经过该柱体最近点时的估计误差。

这里的“可见”仅指满足距离和完整视场限制；`truth_angle` 不处理柱体间遮挡。

### 13.9 真值驱动 + 分割图旁路诊断

保持现有 PF、planner 和 PID 参数，用真值角度继续避障，同时运行独立的诊断进程：

```bash
source /home/bricsbot/catkin_ws/scripts/setup_stonefish_ros.sh
roslaunch aquaflow_stonefish bearing_pf_stonefish.launch \
  sensor_mode:=truth_angle \
  shadow_compare:=true \
  enable:=true \
  rviz:=true \
  log_dir:=/tmp/bearing_pf_truth_shadow
```

新脚本是 `aquaflow_stonefish/scripts/bearing_pf_shadow_node.py`。源码工作区中可直接启动；如果使用 catkin 的安装空间，需要先重新编译/安装，使新增脚本可被找到。

诊断节点只订阅分割图、相机内参和 odometry，读取当前真值模式的几何及检测参数。它没有路径、速度或推进器发布器，也不调用 PF/planner。图像缺失、配对失败或诊断进程退出不会改变真值观测的输入路径；诊断仍会占用少量 CPU/内存。

每次启动在 `log_dir` 下创建独立的 `shadow_日期_时间` 子目录，避免覆盖历史诊断数据；完整位置会打印到终端。不指定 `log_dir` 时使用 `/tmp/bearing_pf_shadow`。

| 文件 | 内容 |
| --- | --- |
| `metadata.json` | 实际采用的场景、柱体标签/几何、相机偏移、检测阈值、配对阈值和误差定义 |
| `frames.csv` | 每个处理帧的图像与内参时间戳、配对方式、最近时间差、插值间隔、机体 yaw/roll/pitch、原始标签列表，以及未配对/无内参等状态 |
| `angle_comparison.csv` | 每个成功配对帧、每根配置柱体的真值角度、mask 角度、像素框、完整/部分/拒绝/缺失状态和误差 |
| `summary.json` | 每 2 秒和退出时更新：有效帧数、跳过/丢帧计数、检测状态、每 ID 的角度偏差、MAE、RMSE 和最大绝对误差（度） |

误差统一为 **分割角度 − 真值角度**，bearing 差包裹到 `[-π, π)`。CSV 角度单位为弧度。未检测或不被适配器接收的观测，其正式误差字段留空，绝不当作零误差；`bbox_*_error_rad` 另外记录实际可见框的偏差，便于排查残缺轮廓。

状态解释：

- `truth_available=1`：与驱动用 oracle 的视场/距离门控一致，不表示无遮挡。
- `segmentation_status=accepted_complete`：现有适配器把它当作完整角宽接收，不保证真实轮廓完整。`partial` 表示适配器只输出禁行角扇区；`rejected`、`missing` 分别表示被拒绝、有此 ID 的像素完全缺失。
- `predicted_occlusion`：按每根柱体 129 条水平射线推算 `unoccluded`、`partial_occlusion`、`occluded`；是柱体间遮挡的二维近似，不包含池壁、自身遮挡和 roll/pitch。
- `summary.errors.accepted_*` 汇总全部被接收的观测；`unoccluded_full_fov_*` 单独汇总预计无遮挡且完整入镜的观测，更适合判断固定角度偏差。
- `unexpected_labels` 包括池壁等非柱体标签，不直接代表误检；可用它检查柱体 ID 配置是否匹配当前场景。

配对使用原始图像 header 时间戳。优先对前后 odometry 插值，yaw 按最短角差插值；前后样本间隔超过 `shadow_max_interpolation_gap_s=0.10` 就拒绝插值。最多等待 0.20 秒接收时间后，若没有后续 odometry，只允许最近样本在 `shadow_max_pair_skew_s=0.05` 内作为明确标记的 `nearest` 配对。零/重复/倒退图像时间戳、非 `world_ned` odometry、队列溢出均单独计数。该配对不能修正 Stonefish 将图像交付时间当成曝光时间的误差，因此应结合 `image_age_s`、`pair_mode` 和机器人转向一起检查。

RViz 新增 **Truth vs Segmentation** 图层：橙色射线是真值方向和角宽，紫色射线是同一帧分割框方向和角宽，长度固定为 1.5 m，并不表示真实距离。原来的 PF 估计可视化继续来自真值驱动的 PF。

实时检查：

```bash
rostopic echo -n 1 /bearing_pf/shadow_status
```

本入口的真值参数对应固定 `bearing_pf_pillars` 场景，诊断节点会拒绝其他场景名，避免拿错误几何作对照。切换到新的仿真时间轴时应重新启动诊断，保留独立日志。

无 ROS 的验证入口（包含 ROS 消息替身的节点连接/日志测试，不等同于真实仿真）：

```bash
cd /home/bricsbot/catkin_ws
.venv-hofa-rl/bin/python -m unittest discover \
  -s src/bearing_sim/bearing_angle_avoidance_demo/bearing_avoidance \
  -p 'test_*shadow*.py'
```

### 13.10 常见启动错误

如果出现 `Unable to communicate with master`，先确认当前终端和启动 Stonefish 的终端使用同一个 ROS master：

```bash
source /home/bricsbot/catkin_ws/scripts/setup_stonefish_ros.sh
export ROS_MASTER_URI=http://localhost:11311
export ROS_HOSTNAME=localhost
rosnode list
```

若 `rosnode list` 仍无法连接，先在一个终端运行 `roscore`，再在另一个终端运行 `roslaunch` 和 `rosrun`。不要只在没有 ROS 环境变量的普通 shell 中执行 `rostopic`。

`inspect_segmentation_labels.py` 直接导入共享的 ROS-free 观测适配器，不应从 `bearing_pf_stonefish_node` 导入 `CameraModel`。若仍看到旧的导入栈，重新执行一次：

```bash
cd /home/bricsbot/catkin_ws
catkin_make --pkg aquaflow_stonefish
source /home/bricsbot/catkin_ws/scripts/setup_stonefish_ros.sh
```

### 13.11 密集随机柱场景

`scripts/generate_bearing_pf_scenes.py` 基于 P1 场景（水池、光照、机器人不变）只替换柱子，默认生成 5 组（seed 101–105），每组 15 根、直径 0.30–0.60 m、柱面最小间隙 0.75 m，至少 3 根挡在起点–终点直线上。只保留膨胀 `robot_radius+safety_margin` 后在走廊内仍有通路的布局。输出在 `src/aquaflow_stonefish/scenarios/bearing_pf_dense/`：`.scn` 场景、`.yaml` 真值参数、`.png` 俯视预览（x 向右、y 向下，粉圈为膨胀半径）。

```bash
python3 scripts/generate_bearing_pf_scenes.py --seeds 101 102 103 104 105 --count 15
```

RGB 闭环（yaml 在 rgb 模式下只用于 RViz 真值层，不进入关联、PF 或规划器）：

```bash
S=dense_s101; LOG=/home/bricsbot/catkin_ws/results/bearing_pf/rgb_${S}_$(date +%m%d_%H%M); mkdir -p $LOG
roslaunch aquaflow_stonefish bearing_pf_stonefish.launch sensor_mode:=rgb \
  scenario_path:=$(rospack find aquaflow_stonefish)/scenarios/bearing_pf_dense/$S.scn \
  truth_pillars_file:=$(rospack find aquaflow_stonefish)/scenarios/bearing_pf_dense/$S.yaml \
  image_topic:=/bricsbot/camera_front/image_color camera_info_topic:=/bricsbot/camera_front/camera_info \
  planner_fov_deg:=80.0 target_depth_m:=1.0 enable:=true motion_enabled:=true rviz:=true \
  log_dir:=$LOG 2>&1 | tee $LOG/terminal.log
```

真值对照另加 `sensor_mode:=truth_angle truth_pillars_file:=.../$S.yaml`。离线评价必须指定同一场景：

```bash
python3 scripts/evaluate_bearing_pf_stonefish_run.py $LOG \
  --scene src/aquaflow_stonefish/scenarios/bearing_pf_dense/$S.scn
```

分割模式的柱 ID 假定沿用 P1 规律（12 起、步长 2），使用前先用 `inspect_segmentation_labels.py` 确认。

#### 第一轮密集场景（`rgb_dense_s101`–`s105`，10-29）问题与修复

| 场景 | 现象 | 原因 | 修复 |
|---|---|---|---|
| s101 | 离终点 0.5 m 处停 28 s，未到达 | 柱 01 只在 29.8 s 被看到一次（`rgb_15`），随后离开视野；单帧后验沿射线拉长（std_x 0.54 m），远端粒子覆盖停车位置，hold 风险 0.11–0.17 > 0.08，规划器、宽视野转向和后退全部不可行 | 停车点风险超阈值时 `recover_escape`：每张地图只在线段比当前净空更近（或从无碰撞变为碰撞）时计为碰撞，选 36 个方向中进度代价最小者，视野外先原地转向 |
| s102/s103/s104 | 中途反复 `cost_prefers_hold`/`no_feasible`，s103 在 (-2.58,-0.91) 停约 9 s | 57–79 % 的 partial sector 宽度 < 1°（2–3 px），多数紧贴另一柱的边缘：柱子自身轮廓的明暗/抗锯齿让 `_split_run` 切出一条“被遮挡”细条，每条按 ±8° 封锁航向；停车期间可用航向从约 10.5 降到 4.5。中间偶尔一帧 0.12 m/s 的 `moving` 使旧的连续计时器每次清零，恢复从未触发 | `split_min_piece_px=5`：不产生窄于 5 px 的切片；卡住判定改为窗口：最近 `stuck_timeout_s` 内 ≥ `stuck_hold_fraction` 的帧在停且位移 ≤ `stuck_max_progress_m`，触发后持续到规划器自行运动 |
| s105 | 与柱 07 擦碰（净空 −0.004 m） | 柱 07 在 16–20 s 始终被图像右边缘截断（可见宽 10–25°，距离 0.6–1 m），从未建立航迹；只靠固定 ±8° 余量，沿切线 ~15° 方向通过时侧向净空不足 | `pad_edge_sectors`：被图像边缘截断的扇区内侧余量按 `sin(θ/2+Δ) ≥ (r+R+m)·sin(θ/2)/r` 计算，θ 取可见宽度、r 取 `partial_edge_min_radius_m=0.15` |

s104/s105 结尾在终点 0.27 m 处的“慢”是 `goal_reached` 后的保持，不是卡住。测试：`test_stonefish_recovery.py`（卡住窗口、逃逸、边缘余量），`test_stonefish_observation.py` 新增轮廓细条用例。

#### 第二轮：`rgb_dense_s101_1029_1646` 碰撞与截断轮廓航迹

43.2 s 到达终点，但 38.8 s 在 (3.57, 0.32) 与 BearingPillar01 (3.476, −0.112, r 0.268) 擦碰，净空 −0.021 m。柱 01 在 29.9–37.7 s 一直被图像左侧（bearing −40°，u=0）截断，距离 1.7→0.37 m，真实角宽 18°→90°，可见扇区却只有约 −40:−25°；从未建立航迹，38 s 离开视野后规划器里不再有它。`pad_edge_sectors` 用可见宽度（≈15°）只加出 ≈15° 余量，实际需要离中心约 64°：可见宽度严重低估截断柱，这一启发式不保守。

修复：被**一侧**图像边缘截断、且不是切分产生的遮挡片（`ColorComponent.occluded`）的分量，内侧边界仍是真实切线。用截断观测 `z=[edge, bound, side]` 更新 PF（`bearing_angle_pf.log_likelihood`）：近切线 `beta − side·theta/2` 对 `edge` 用 σβ 高斯似然，远切线需越过图像边界 `bound`，否则按越不过的量惩罚。多个位姿的切线射线三角化出中心和直径；离开视野后航迹保留，规划器仍有记忆。`side=+1` 表示柱子向更大 bearing 方向延伸（被图像右侧，即 +half_fov 截断），`−1` 相反。

- `RgbAssociator`：对已有航迹用 `predict_edge` 预测内侧切线，用与完整检测相同的门限匹配。未匹配、可见宽度在 `rgb_min_new_edge_width_deg`（2°）到 `rgb_max_new_edge_width_deg`（60°）之间的截断分量新建航迹。截断扇区同时仍作为 partial sector 交给规划器。
- 首帧截断初始化 `_initialize_truncated`：在先验中限制 `L ≥ 2d·sin(w/2)`。
- 观测日志 `bearing_rad/width_rad` 对截断观测记录可见扇区，新列 `truncated_side`（0 = 完整）。

离线回放 1646 记录的扇区与完整检测：柱 01 从 29.7 s 的第一帧截断开始建航迹，0.5 s 内中心误差 0.04 m，最终 0.02 m、L 0.57（真值 0.535），碰撞位姿被膨胀圆覆盖。已知限制：未被切分的两柱合并暗区若贴边，内侧边缘属于较远那根柱，会拟合出位于两柱之间的“幻影”圆（s104 回放 t=0 的柱 15/14）。它在真实遮挡区内，偏保守。测试：`test_rgb_association.py::edge_track_s101`。
