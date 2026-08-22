"""Coordinate frame conversions between NED/FRD and ENU/FLU conventions.

Internal algorithm uses ENU/FLU.  The simulator adapter layer converts
to/from the simulator's native convention (typically NED/FRD).
"""
import numpy as np


def ned_to_enu_position(pos_ned: np.ndarray) -> np.ndarray:
    """Convert NED position to ENU: x_enu=y_ned, y_enu=x_ned, z_enu=-z_ned."""
    return np.array([pos_ned[1], pos_ned[0], -pos_ned[2]])


def enu_to_ned_position(pos_enu: np.ndarray) -> np.ndarray:
    """Convert ENU position to NED."""
    return np.array([pos_enu[1], pos_enu[0], -pos_enu[2]])


def ned_to_enu_yaw(yaw_ned: float) -> float:
    """Convert NED yaw to ENU yaw.  NED yaw is clockwise from north;
    ENU yaw is counter-clockwise from east.  The mapping is: yaw_enu = pi/2 - yaw_ned.
    Wraps result to [-pi, pi]."""
    return np.arctan2(np.sin(np.pi / 2 - yaw_ned),
                      np.cos(np.pi / 2 - yaw_ned))


def enu_to_ned_yaw(yaw_enu: float) -> float:
    """Convert ENU yaw to NED yaw: yaw_ned = pi/2 - yaw_enu."""
    return np.arctan2(np.sin(np.pi / 2 - yaw_enu),
                      np.cos(np.pi / 2 - yaw_enu))


def ned_to_enu_velocity(vel_ned: np.ndarray, yaw_ned: float) -> np.ndarray:
    """Convert NED velocity (x_ned, y_ned, r) to ENU (x_enu, y_enu, r).

    For linear velocity: v_enu = R(90deg) * v_ned (swap and negate z is 2D).
    For angular velocity: r_enu = -r_ned (NED CW positive, ENU CCW positive).
    """
    v_enu = np.array([vel_ned[1], vel_ned[0]])
    r_enu = -vel_ned[2]
    return np.array([v_enu[0], v_enu[1], r_enu])


def enu_to_ned_velocity(vel_enu: np.ndarray) -> np.ndarray:
    """Convert ENU velocity to NED velocity."""
    v_ned = np.array([vel_enu[1], vel_enu[0]])
    r_ned = -vel_enu[2]
    return np.array([v_ned[0], v_ned[1], r_ned])


def ned_to_enu_odom(pose_ned: np.ndarray, vel_ned_linear: np.ndarray,
                    yaw_rate_ned: float):
    """Convert full NED odometry tuple to ENU.

    Args:
        pose_ned: [x, y, z, roll, pitch, yaw] in NED
        vel_ned_linear: [vx, vy, vz] in NED body or world frame
        yaw_rate_ned: yaw rate in NED

    Returns:
        (pos_enu, yaw_enu, vel_enu_3d) where vel_enu_3d = [vx, vy, r]
    """
    pos_enu = ned_to_enu_position(pose_ned[:3])
    yaw_enu = ned_to_enu_yaw(pose_ned[5])
    vel_2d_ned = np.array([vel_ned_linear[0], vel_ned_linear[1], yaw_rate_ned])
    vel_enu = ned_to_enu_velocity(vel_2d_ned, pose_ned[5])
    return pos_enu, yaw_enu, vel_enu


def body_to_world(psi: float, vec_body: np.ndarray) -> np.ndarray:
    """Rotate a 2D vector from body frame to world frame."""
    c, s = np.cos(psi), np.sin(psi)
    R = np.array([[c, -s], [s, c]])
    return R @ vec_body


def world_to_body(psi: float, vec_world: np.ndarray) -> np.ndarray:
    """Rotate a 2D vector from world frame to body frame."""
    c, s = np.cos(psi), np.sin(psi)
    R = np.array([[c, s], [-s, c]])
    return R @ vec_world


def shortest_angle_error(target: float, current: float) -> float:
    """Compute shortest angular error from current to target, wrapped to [-pi, pi]."""
    return np.arctan2(np.sin(target - current), np.cos(target - current))
