#!/usr/bin/env python3
"""HOFA-MPC controller node for BricsBot.

Subscribes to Stonefish NED odometry, runs MPC in ENU internally,
outputs body-frame generalized force compatible with brics6_thruster_allocator.
"""
import math
import time
import numpy as np
import rospy
import tf.transformations as tft
from std_msgs.msg import Header, Bool, Empty
from geometry_msgs.msg import WrenchStamped, AccelStamped, PoseStamped, Quaternion
from nav_msgs.msg import Odometry, Path
from hofa_mpc_ros.msg import TrajectoryPoint, TrajectoryPointWindow, ControllerStatus

from hofa_mpc_ros.types import (
    VehicleParams, MPCParams, VehicleState, ReferencePoint,
    VirtualInputBounds, ControllerState
)
from hofa_mpc_ros.model import ThreeDOFModel
from hofa_mpc_ros.mpc import HofaMPC
from hofa_mpc_ros.hofa import hofa_inverse, wrap_to_pi
from hofa_mpc_ros.constraints import SafeInnerBoxStrategy
from hofa_mpc_ros.coordinates import shortest_angle_error
from hofa_mpc_ros.allocator import ThrusterAllocator
from hofa_mpc_ros.types import ThrusterConfig
from hofa_mpc_ros.safety import SafetySupervisor, SafetyParams


class HofaMPCControllerNode:
    def __init__(self):
        rospy.init_node("hofa_mpc_controller")

        # --- Load parameters ---
        self._load_params()

        # --- Core algorithm objects ---
        self.model = ThreeDOFModel(self.vehicle_params)
        self.mpc = HofaMPC(self.mpc_params, self.vehicle_params)
        self.constraint_strategy = SafeInnerBoxStrategy()
        self.allocator = self._build_allocator()
        self.safety = SafetySupervisor(self.safety_params)

        # --- State ---
        self.state = VehicleState()
        self.state_received = False
        self.state_time = 0.0
        self.ref = None
        self.ref_window = None
        self.ref_time = 0.0
        self.prev_wrench = np.zeros(3)
        self.enabled = rospy.get_param("~enabled", False)
        if self.enabled:
            self.controller_state = ControllerState.WAITING_FOR_STATE
        else:
            self.controller_state = ControllerState.DISABLED

        # --- Subscribers ---
        odom_topic = rospy.get_param("~odom_topic", "/bricsbot/odometry")
        self.odom_sub = rospy.Subscriber(
            odom_topic, Odometry, self._odom_cb, queue_size=10)

        self.enable_sub = rospy.Subscriber(
            "~enable", Bool, self._enable_cb, queue_size=1)
        self.reset_sub = rospy.Subscriber(
            "~reset", Empty, self._reset_cb, queue_size=1)

        # Reference trajectory from reference_processor (arc-length projected)
        self.traj_ref_sub = rospy.Subscriber(
            "/controller/reference_trajectory", TrajectoryPoint,
            self._traj_ref_cb, queue_size=1)
        self.traj_window_sub = rospy.Subscriber(
            "/controller/reference_trajectory_window", TrajectoryPointWindow,
            self._traj_window_cb, queue_size=1)

        # --- Publishers ---
        wrench_topic = rospy.get_param(
            "~wrench_topic", "/controller/generalized_force")
        self.wrench_pub = rospy.Publisher(
            wrench_topic, WrenchStamped, queue_size=5)
        self.status_pub = rospy.Publisher(
            "~status", ControllerStatus, queue_size=5)
        self.virtual_accel_pub = rospy.Publisher(
            "~virtual_accel_cmd", AccelStamped, queue_size=5)
        self.predicted_path_pub = rospy.Publisher(
            "~predicted_path", Path, queue_size=1)
        self.error_x_pub = rospy.Publisher(
            "~tracking_error/x_m", rospy.msg.AnyMsg, queue_size=10)
        self.error_y_pub = rospy.Publisher(
            "~tracking_error/y_m", rospy.msg.AnyMsg, queue_size=10)
        self.error_yaw_pub = rospy.Publisher(
            "~tracking_error/yaw_rad", rospy.msg.AnyMsg, queue_size=10)

        # --- Control timer ---
        rate = self.mpc_params.control_rate_hz
        self.timer = rospy.Timer(rospy.Duration(1.0 / rate), self._control_cb)
        self.last_callback_time = None

        rospy.loginfo("HOFA-MPC controller ready: rate=%.1f Hz, horizon=%d",
                      rate, self.mpc_params.horizon)

    def _load_params(self):
        # Vehicle params
        vp = rospy.get_param("~vehicle", {})
        mass_list = vp.get("mass_matrix", [7.94, 0, 0, 0, 7.94, 0, 0, 0, 0.15])
        self.vehicle_params = VehicleParams(
            mass_matrix=np.array(mass_list).reshape(3, 3),
            drag_linear=np.array(vp.get("drag_linear", [8.0, 10.0, 1.4])),
            drag_quadratic=np.array(vp.get("drag_quadratic", [12.0, 15.0, 0.35])),
            coriolis_enabled=vp.get("coriolis_enabled", False),
        )

        # MPC params
        mp = rospy.get_param("~mpc", {})
        self.mpc_params = MPCParams(
            control_rate_hz=float(mp.get("control_rate_hz", 10.0)),
            horizon=int(mp.get("horizon", 14)),
            weight_pose=np.array(mp.get("weights", {}).get("pose", [55, 55, 32])),
            weight_pose_rate=np.array(mp.get("weights", {}).get("pose_rate", [7, 7, 5.5])),
            weight_virtual_input=np.array(mp.get("weights", {}).get("virtual_input", [0.2, 0.2, 0.14])),
            weight_input_increment=np.array(mp.get("weights", {}).get("input_increment", [1.2, 1.2, 0.75])),
            terminal_multiplier=float(mp.get("weights", {}).get("terminal_multiplier", 4.0)),
            safe_box_scale=float(mp.get("safe_box_scale", 1.0 / 3.0)),
            max_iterations=int(mp.get("solver", {}).get("max_iterations", 90)),
            deadline_ms=float(mp.get("solver", {}).get("deadline_ms", 80.0)),
            warm_start=mp.get("solver", {}).get("warm_start", True),
        )

        # Safety params
        sp = rospy.get_param("~safety", {})
        self.safety_params = SafetyParams()
        self.safety_params.state_timeout_s = float(sp.get("state_timeout_s", 0.20))
        self.safety_params.reference_timeout_s = float(sp.get("reference_timeout_s", 0.30))
        self.safety_params.max_position_abs = np.array(
            sp.get("max_position_abs_m", [14.0, 8.0]))
        self.safety_params.max_world_speed = float(sp.get("max_world_speed_mps", 1.0))
        self.safety_params.max_yaw_rate = float(sp.get("max_yaw_rate_radps", 1.2))
        self.safety_params.max_consecutive_solver_failures = int(
            sp.get("max_consecutive_solver_failures", 3))
        self.safety_params.max_saturation_duration_s = float(
            sp.get("max_saturation_duration_s", 2.0))
        self.safety_params.zero_on_time_jump = bool(
            sp.get("zero_on_time_jump", True))

        # Thruster params
        tp = rospy.get_param("~thrusters", {})
        self.thruster_params = tp

    def _build_allocator(self):
        tp = self.thruster_params
        n = int(tp.get("count", 4))
        positions = [np.array(p) for p in tp.get("positions_xy", [[0,0]]*n)]
        directions = [np.array(d) for d in tp.get("directions_xy", [[1,0]]*n)]
        f_max = tp.get("thrust_max", [67.67]*n)
        f_min = tp.get("thrust_min", [-58.84]*n)

        thrusters = []
        for i in range(n):
            thrusters.append(ThrusterConfig(
                position=positions[i],
                direction=directions[i],
                thrust_max=f_max[i],
                thrust_min=f_min[i],
            ))
        return ThrusterAllocator(thrusters)

    # --- Callbacks ---

    def _odom_cb(self, msg):
        """Handle Stonefish NED odometry -> convert to ENU VehicleState."""
        stamp = msg.header.stamp.to_sec()

        # NED position
        x_ned = msg.pose.pose.position.x
        y_ned = msg.pose.pose.position.y
        z_ned = msg.pose.pose.position.z

        # NED yaw from quaternion
        q = msg.pose.pose.orientation
        yaw_ned = tft.euler_from_quaternion([q.x, q.y, q.z, q.w])[2]

        # Stonefish odometry velocities are already expressed in the sensor
        # (body) frame; see Odometry::InternalUpdate, which applies the
        # inverse sensor basis before publishing.  They must not be treated as
        # world-frame velocities and rotated a second time.
        vx_ned = msg.twist.twist.linear.x
        vy_ned = msg.twist.twist.linear.y
        r_ned = msg.twist.twist.angular.z

        # Convert to ENU
        x_enu = y_ned       # NED-y (east) -> ENU-x (east)
        y_enu = x_ned       # NED-x (north) -> ENU-y (north)
        yaw_enu = math.pi / 2 - yaw_ned  # NED yaw -> ENU yaw
        yaw_enu = wrap_to_pi(yaw_enu)

        # Convert body velocity NED/FRD -> body velocity ENU/FLU.  x is
        # forward in both conventions; y and positive yaw change sign.
        u_enu = vx_ned
        v_enu = -vy_ned
        r_enu = -r_ned

        self.state = VehicleState(
            x=x_enu, y=y_enu, psi=yaw_enu,
            u=u_enu, v=v_enu, r=r_enu,
            timestamp=stamp
        )
        self.state_received = True
        self.state_time = stamp

    def _enable_cb(self, msg):
        self.enabled = msg.data
        if self.enabled:
            self.controller_state = ControllerState.WAITING_FOR_STATE
            rospy.loginfo("Controller ENABLED")
        else:
            self.controller_state = ControllerState.DISABLED
            self._publish_zero_wrench()
            rospy.loginfo("Controller DISABLED")

    def _reset_cb(self, _msg):
        self.mpc.reset()
        self.safety.reset()
        self.prev_wrench = np.zeros(3)
        rospy.loginfo("Controller RESET")

    def _traj_ref_cb(self, msg):
        """Handle TrajectoryPoint from reference_processor (arc-length projected)."""
        q = msg.pose.orientation
        psi = tft.euler_from_quaternion([q.x, q.y, q.z, q.w])[2]

        self.ref = ReferencePoint(
            # Reference processor publishes world_ned.  The controller's
            # internal state/model are ENU/FLU, so convert every derivative,
            # not only position and yaw.
            x=msg.pose.position.y,
            y=msg.pose.position.x,
            psi=wrap_to_pi(math.pi / 2.0 - psi),
            dx=msg.twist.linear.y,
            dy=msg.twist.linear.x,
            dpsi=-msg.twist.angular.z,
            ddx=msg.accel.linear.y,
            ddy=msg.accel.linear.x,
            ddpsi=-msg.accel.angular.z,
        )
        self.ref_time = msg.header.stamp.to_sec()

    def _reference_from_msg(self, msg):
        q = msg.pose.orientation
        psi_ned = tft.euler_from_quaternion([q.x, q.y, q.z, q.w])[2]
        return ReferencePoint(
            x=msg.pose.position.y, y=msg.pose.position.x,
            psi=wrap_to_pi(math.pi / 2.0 - psi_ned),
            dx=msg.twist.linear.y, dy=msg.twist.linear.x,
            dpsi=-msg.twist.angular.z,
            ddx=msg.accel.linear.y, ddy=msg.accel.linear.x,
            ddpsi=-msg.accel.angular.z)

    def _traj_window_cb(self, msg):
        if not msg.valid or not msg.points:
            return
        self.ref_window = [self._reference_from_msg(point) for point in msg.points]
        self.ref = self.ref_window[0]
        self.ref_time = msg.header.stamp.to_sec()

    # --- Control loop ---

    def _control_cb(self, _event):
        now = rospy.Time.now().to_sec()
        t_start = time.time()

        # Check safety
        self.controller_state = self.safety.check_state(
            self.state, now, self.ref_time if self.ref else None)

        if not self.enabled:
            self.controller_state = ControllerState.DISABLED

        # Get override command from safety supervisor
        override, effective_state = self.safety.get_override_command(
            self.controller_state)

        if override is not None:
            self._publish_wrench_ned(override)
            self.controller_state = effective_state
            self._publish_status(0, 0, 0, t_start)
            return

        # Need both state and reference
        if not self.state_received or self.ref is None:
            self._publish_zero_wrench()
            self._publish_status(0, 0, 0, t_start)
            return

        # Use the complete local-planner window.  The fallback keeps the node
        # compatible with old publishers, but no longer invents a horizon when
        # the window topic is available.
        dt = 1.0 / self.mpc_params.control_rate_hz
        refs = list(self.ref_window or [])
        if not refs:
            refs = [ReferencePoint(
                x=self.ref.x + self.ref.dx * i * dt,
                y=self.ref.y + self.ref.dy * i * dt,
                psi=wrap_to_pi(self.ref.psi + self.ref.dpsi * i * dt),
                dx=self.ref.dx, dy=self.ref.dy, dpsi=self.ref.dpsi,
                ddx=self.ref.ddx, ddy=self.ref.ddy, ddpsi=self.ref.ddpsi)
                for i in range(self.mpc_params.horizon)]
        if len(refs) < self.mpc_params.horizon:
            refs.extend([refs[-1]] * (self.mpc_params.horizon - len(refs)))
        refs = refs[:self.mpc_params.horizon]

        # Compute virtual input bounds (Layer 1)
        t_layer1_start = time.time()
        bounds = self.constraint_strategy.compute(
            self.state, self.model, self.allocator,
            scale=self.mpc_params.safe_box_scale,
        )
        t_layer1_ms = (time.time() - t_layer1_start) * 1000

        # Solve MPC (Layer 2)
        t_layer2_start = time.time()
        sol = self.mpc.solve(self.state, refs, bounds=bounds)
        t_layer2_ms = (time.time() - t_layer2_start) * 1000

        if not sol.success:
            self.safety.on_solver_result(False, np.zeros(3))
            override, effective_state = self.safety.get_override_command(
                self.controller_state)
            if override is not None:
                self._publish_wrench_ned(override)
                self.controller_state = effective_state
                return

        # HOFA inverse: virtual acceleration -> body force (ENU/FLU)
        state_arr = self.state.to_array()
        wrench_enu = hofa_inverse(state_arr, sol.virtual_accel, self.model)

        # Apply generalized force scale (positive/negative per axis)
        gfs = rospy.get_param("~generalized_force_scale", {})
        scale = np.array([
            gfs.get("x_positive", 1.0) if wrench_enu[0] >= 0 else gfs.get("x_negative", 1.0),
            gfs.get("y_positive", 1.0) if wrench_enu[1] >= 0 else gfs.get("y_negative", 1.0),
            gfs.get("yaw_positive", 1.0) if wrench_enu[2] >= 0 else gfs.get("yaw_negative", 1.0),
        ])
        wrench_enu = wrench_enu * scale

        # Clamp
        wrench_enu = self.safety.validate_wrench(wrench_enu)
        # Store the actual force command for degraded-mode hold.  The solver's
        # MPCSolution.wrench field is not populated by the optimizer.
        self.safety.on_solver_result(True, wrench_enu)

        # Convert FLU body force -> FRD body force for Stonefish
        # FLU: x-forward, y-left, z-up
        # FRD: x-forward, y-right, z-down
        # Fx same, Fy negate, N same, Fz=0
        wrench_ned = np.array([wrench_enu[0], -wrench_enu[1], wrench_enu[2]])

        # Publish
        self._publish_wrench_ned(wrench_ned)
        self._publish_virtual_accel(sol.virtual_accel)
        self._publish_predicted_path(sol.predicted_path, refs)
        self.prev_wrench = wrench_ned

        # Status
        total_ms = (time.time() - t_start) * 1000
        pos_err = math.hypot(self.state.x - self.ref.x, self.state.y - self.ref.y)
        yaw_err = abs(shortest_angle_error(self.ref.psi, self.state.psi))
        self._publish_status(pos_err, yaw_err, total_ms,
                             t_start, t_layer1_ms, t_layer2_ms,
                             sol.success, sol.iterations, sol.objective)

    def _publish_wrench_ned(self, wrench_ned):
        """Publish generalized force in NED/FRD body frame."""
        msg = WrenchStamped()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = "base_link"
        msg.wrench.force.x = float(wrench_ned[0])
        msg.wrench.force.y = float(wrench_ned[1])
        msg.wrench.torque.z = float(wrench_ned[2])
        msg.wrench.force.z = 0.0  # depth handled separately
        self.wrench_pub.publish(msg)

    def _publish_zero_wrench(self):
        self._publish_wrench_ned(np.zeros(3))

    def _publish_virtual_accel(self, accel):
        msg = AccelStamped()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = "world_ned"
        msg.accel.linear.x = float(accel[0])
        msg.accel.linear.y = float(accel[1])
        msg.accel.angular.z = float(accel[2])
        self.virtual_accel_pub.publish(msg)

    def _publish_predicted_path(self, predicted, refs):
        path = Path()
        path.header.stamp = rospy.Time.now()
        path.header.frame_id = "world_ned"
        for i in range(len(predicted)):
            ps = PoseStamped()
            ps.header = path.header
            # Convert ENU predicted position back to NED for visualization
            ps.pose.position.x = predicted[i, 1]   # ENU-y -> NED-x
            ps.pose.position.y = predicted[i, 0]    # ENU-x -> NED-y
            ps.pose.position.z = 0.0
            psi = predicted[i, 2]
            yaw_ned = math.pi / 2 - psi
            q = tft.quaternion_from_euler(0, 0, yaw_ned)
            ps.pose.orientation = Quaternion(x=q[0], y=q[1], z=q[2], w=q[3])
            path.poses.append(ps)
        self.predicted_path_pub.publish(path)

    def _publish_status(self, pos_err, yaw_err, callback_ms, t_start,
                        layer1_ms=0, layer2_ms=0, success=True,
                        iterations=0, objective=0.0):
        msg = ControllerStatus()
        msg.header.stamp = rospy.Time.now()
        msg.state = self.controller_state.value
        msg.solver_success = success
        msg.solver_iterations = iterations
        msg.objective_value = objective
        msg.layer1_time_ms = layer1_ms
        msg.layer2_time_ms = layer2_ms
        msg.callback_time_ms = callback_ms
        msg.position_error_m = pos_err
        msg.yaw_error_rad = yaw_err
        self.status_pub.publish(msg)

    def shutdown(self):
        self._publish_zero_wrench()
        rospy.loginfo("HOFA-MPC controller shut down")


if __name__ == "__main__":
    try:
        node = HofaMPCControllerNode()
        rospy.on_shutdown(node.shutdown)
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
