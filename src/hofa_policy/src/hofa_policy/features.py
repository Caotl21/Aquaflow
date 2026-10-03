"""Causal NED/FRD feature construction shared by export and inference."""
import math

import numpy as np

from .contract import CONTRACT


def wrap_to_pi(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


def reference_features(odom, reference_rows, horizon=CONTRACT.horizon):
    """Convert a NED reference window into current body-FRD features."""
    if len(reference_rows) < horizon:
        raise ValueError("reference window is shorter than policy horizon")
    yaw = float(odom["yaw"])
    c, s = math.cos(yaw), math.sin(yaw)
    output = np.zeros((horizon, CONTRACT.reference_features),
                      dtype=np.float32)
    for i, ref in enumerate(reference_rows[:horizon]):
        dx = float(ref["x_ned"]) - float(odom["x"])
        dy = float(ref["y_ned"]) - float(odom["y"])
        x_body = c * dx + s * dy
        y_body = -s * dx + c * dy
        delta_yaw = wrap_to_pi(float(ref["yaw_ned"]) - yaw)
        vx_body = c * float(ref["dx_ned"]) + s * float(ref["dy_ned"])
        vy_body = -s * float(ref["dx_ned"]) + c * float(ref["dy_ned"])
        ax_body = c * float(ref["ddx_ned"]) + s * float(ref["ddy_ned"])
        ay_body = -s * float(ref["ddx_ned"]) + c * float(ref["ddy_ned"])
        output[i] = [
            x_body, y_body, math.sin(delta_yaw), math.cos(delta_yaw),
            vx_body, vy_body, float(ref["dpsi_ned"]),
            ax_body, ay_body, float(ref["ddpsi_ned"]),
        ]
    return output


def state_features(odom):
    return np.asarray([odom["u"], odom["v"], odom["r"]],
                      dtype=np.float32)


def horizontal_wrench(row):
    return np.asarray([row["fx"], row["fy"], row["nz"]],
                      dtype=np.float32)
