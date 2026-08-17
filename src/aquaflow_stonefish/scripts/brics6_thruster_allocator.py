#!/usr/bin/env python3
"""Six-thruster allocator for the lab BricsBot configuration.

The allocator works in force space first, using the measured installation
directions and moment arms, then maps each requested thrust to an asymmetric
26 V PWM limit from the propulsion test sheet.
"""
import math

import rospy
from geometry_msgs.msg import WrenchStamped
from std_msgs.msg import Float64MultiArray


def invert4(matrix):
    """Gauss-Jordan inverse for a 4x4 matrix, avoiding a NumPy dependency."""
    augmented = [list(row) + [1.0 if i == j else 0.0 for j in range(4)]
                 for i, row in enumerate(matrix)]
    for col in range(4):
        pivot = max(range(col, 4), key=lambda row: abs(augmented[row][col]))
        if abs(augmented[pivot][col]) < 1e-12:
            raise ValueError("singular six-thruster wrench matrix")
        augmented[col], augmented[pivot] = augmented[pivot], augmented[col]
        scale = augmented[col][col]
        augmented[col] = [value / scale for value in augmented[col]]
        for row in range(4):
            if row == col:
                continue
            factor = augmented[row][col]
            augmented[row] = [augmented[row][j] - factor * augmented[col][j]
                              for j in range(8)]
    return [row[4:] for row in augmented]


class Brics6Allocator:
    # Columns are T1..T6.  Directions and r_perp are the user's measured
    # body-frame configuration; all distances are metres and NED z is down.
    directions = (
        (0.707, 0.707, 0.0),
        (-0.707, 0.707, 0.0),
        (-0.707, 0.707, 0.0),
        (0.707, 0.707, 0.0),
        (0.0, 0.0, -1.0),
        (0.0, 0.0, 1.0),
    )
    positions = (
        (0.1278, -0.1278, 0.0),
        (0.1273, 0.1273, 0.0),
        (-0.1278, -0.1278, 0.0),
        (-0.1273, 0.1273, 0.0),
        (0.0, -0.1235, 0.0),
        (0.0, 0.1235, 0.0),
    )

    def __init__(self):
        self.vehicle_name = rospy.get_param("~vehicle_name", "bricsbot")
        self.enabled = bool(rospy.get_param("~enabled", False))
        self.timeout = float(rospy.get_param("~wrench_timeout", 0.25))
        # 26 V sheet maxima: +6.9 kgf and -6.0 kgf.
        self.forward_max_force = float(rospy.get_param("~forward_max_force_n", 6.9 * 9.80665))
        self.reverse_max_force = float(rospy.get_param("~reverse_max_force_n", 6.0 * 9.80665))
        self.last_wrench = None
        self.pinv = self._build_pseudoinverse()
        topic = "/%s/setpoint/pwm" % self.vehicle_name
        self.pub = rospy.Publisher(topic, Float64MultiArray, queue_size=1)
        rospy.Subscriber("/controller/generalized_force", WrenchStamped,
                         self.wrench_cb, queue_size=1)
        self.timer = rospy.Timer(rospy.Duration(0.05), self.update)
        rospy.on_shutdown(self.shutdown)
        rospy.loginfo("BricsBot six-thruster allocator ready: +%.2f/-%.2f N",
                      self.forward_max_force, self.reverse_max_force)

    def _build_pseudoinverse(self):
        matrix = [[0.0] * 6 for _ in range(4)]
        for i, (d, r) in enumerate(zip(self.directions, self.positions)):
            dx, dy, dz = d
            rx, ry, _ = r
            matrix[0][i] = dx
            matrix[1][i] = dy
            matrix[2][i] = rx * dy - ry * dx
            matrix[3][i] = dz
        gram = [[sum(matrix[row][i] * matrix[col][i] for i in range(6))
                  for col in range(4)] for row in range(4)]
        gram_inv = invert4(gram)
        # A^T(AA^T)^-1: force solution with minimum squared actuator force.
        return [[sum(matrix[i][row] * gram_inv[i][col] for i in range(4))
                 for col in range(4)] for row in range(6)]

    def wrench_cb(self, msg):
        self.last_wrench = msg

    def force_to_pwm(self, force):
        limit = self.forward_max_force if force >= 0.0 else self.reverse_max_force
        return max(-1.0, min(1.0, force / max(limit, 1e-6)))

    def update(self, _event):
        out = [0.0] * 6
        if self.enabled and self.last_wrench is not None:
            age = (rospy.Time.now() - self.last_wrench.header.stamp).to_sec()
            if 0.0 <= age <= self.timeout:
                w = self.last_wrench.wrench
                wrench = (w.force.x, w.force.y, w.torque.z, w.force.z)
                forces = [sum(self.pinv[i][j] * wrench[j] for j in range(4))
                          for i in range(6)]
                out = [self.force_to_pwm(force) for force in forces]
        self.pub.publish(Float64MultiArray(data=out))

    def shutdown(self):
        self.enabled = False
        self.pub.publish(Float64MultiArray(data=[0.0] * 6))


if __name__ == "__main__":
    rospy.init_node("brics6_thruster_allocator")
    Brics6Allocator()
    rospy.spin()
