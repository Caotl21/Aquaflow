#!/usr/bin/env python3
"""Six-thruster allocator for the lab BricsBot configuration.

The allocator works in force space first, using the measured installation
directions and moment arms, then maps each requested thrust to an asymmetric
26 V PWM limit from the propulsion test sheet.
"""
import bisect
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

    # 26 V propulsion-sheet calibration, expressed as normalized rotor
    # command magnitude -> positive thrust magnitude.  The measured table
    # has a real startup dead zone; direct force/max-force scaling therefore
    # produces commands which only spin the propellers without generating
    # useful thrust.
    positive_curve = (
        (0.0, 0.0), (0.052631579, 0.0),
        (0.105263158, 1.961330), (0.157894737, 3.922660),
        (0.210526316, 7.845320), (0.263157895, 11.767980),
        (0.315789474, 16.671305), (0.368421053, 20.593965),
        (0.421052632, 26.477955), (0.473684211, 32.361945),
        (0.526315789, 37.265270), (0.578947368, 44.129925),
        (0.631578947, 48.052585), (0.684210526, 53.936575),
        (0.736842105, 57.859235), (0.789473684, 63.743225),
        (0.842105263, 64.723890), (0.894736842, 64.723890),
        (0.947368421, 67.665885), (1.0, 67.665885),
    )
    negative_curve = (
        (0.0, 0.0), (0.052631579, 0.0), (0.105263158, 0.0),
        (0.157894737, 0.980665), (0.210526316, 2.941995),
        (0.263157895, 4.903325), (0.315789474, 7.845320),
        (0.368421053, 10.787315), (0.421052632, 14.709975),
        (0.473684211, 18.632635), (0.526315789, 23.535960),
        (0.578947368, 28.439285), (0.631578947, 34.323275),
        (0.684210526, 38.245935), (0.736842105, 43.149260),
        (0.789473684, 48.052585), (0.842105263, 54.917240),
        (0.894736842, 56.878570), (0.947368421, 58.839900),
        (1.0, 58.839900),
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
        # T2, T4, T6 have inverted_setpoint="true" in the Stonefish scenario,
        # meaning Stonefish negates the PWM command before applying the thrust
        # model.  We must negate the allocator output for those thrusters so the
        # actual force matches the direction the allocator computed.
        inverted = rospy.get_param("~inverted_setpoints", [False, True, False, True, False, True])
        self.sign = [(-1.0 if inv else 1.0) for inv in inverted]
        topic = "/%s/setpoint/pwm" % self.vehicle_name
        self.pub = rospy.Publisher(topic, Float64MultiArray, queue_size=1)
        rospy.Subscriber("/controller/generalized_force", WrenchStamped,
                         self.wrench_cb, queue_size=1)
        self.timer = rospy.Timer(rospy.Duration(0.05), self.update)
        rospy.on_shutdown(self.shutdown)
        rospy.loginfo("BricsBot six-thruster allocator ready: +%.2f/-%.2f N  inverted=%s",
                      self.forward_max_force, self.reverse_max_force, inverted)

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

    @staticmethod
    def _inverse_curve(force_abs, curve):
        """Return command magnitude for a calibrated thrust magnitude."""
        if force_abs <= 0.05:
            return 0.0
        forces = [item[1] for item in curve]
        if force_abs >= forces[-1]:
            return curve[-1][0]
        hi = bisect.bisect_left(forces, force_abs)
        lo = max(0, hi - 1)
        c0, f0 = curve[lo]
        c1, f1 = curve[hi]
        if f1 <= f0:
            return c1
        return c0 + (force_abs - f0) * (c1 - c0) / (f1 - f0)

    def force_to_pwm(self, force):
        """Map desired signed force through the measured nonlinear curve.

        The scenario's inverted_setpoint flags make positive allocator force
        correspond to positive force along each configured direction, so the
        allocator only needs the asymmetric magnitude curves here.
        """
        if abs(force) <= 0.05:
            return 0.0
        curve = self.positive_curve if force > 0.0 else self.negative_curve
        command = self._inverse_curve(abs(force), curve)
        return max(-1.0, min(1.0, command if force > 0.0 else -command))

    def update(self, _event):
        out = [0.0] * 6
        if self.enabled and self.last_wrench is not None:
            age = (rospy.Time.now() - self.last_wrench.header.stamp).to_sec()
            if 0.0 <= age <= self.timeout:
                w = self.last_wrench.wrench
                wrench = (w.force.x, w.force.y, w.torque.z, w.force.z)
                forces = [sum(self.pinv[i][j] * wrench[j] for j in range(4))
                          for i in range(6)]
                out = [self.force_to_pwm(force) * self.sign[i]
                       for i, force in enumerate(forces)]
        self.pub.publish(Float64MultiArray(data=out))

    def shutdown(self):
        self.enabled = False
        self.pub.publish(Float64MultiArray(data=[0.0] * 6))


if __name__ == "__main__":
    rospy.init_node("brics6_thruster_allocator")
    Brics6Allocator()
    rospy.spin()
