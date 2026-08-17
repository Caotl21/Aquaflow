#!/usr/bin/env python3
"""Allocate [Fx,Fy,Nz] to the four horizontal Stonefish thrusters."""
import rospy
from geometry_msgs.msg import WrenchStamped
from std_msgs.msg import Float64MultiArray


class Blue8Allocator:
    def __init__(self):
        self.last_wrench = None
        self.enabled = bool(rospy.get_param("~enabled", False))
        self.timeout = float(rospy.get_param("~wrench_timeout", 0.25))
        self.force_per_pwm = float(rospy.get_param("~force_per_normalized_pwm", 0.8))
        # Order is the actuator order in bluerov2_reference.scn. The final
        # Channels 5-8 are the four vertical thrusters. Their local +x axis
        # is rotated to +z (NED down), and their configured signs make a
        # positive PWM produce positive body/world_ned z force.
        # Input-to-body wrench matrix after Stonefish's inverted_setpoint and
        # right/left propeller conventions in bluerov2_reference.scn. The
        # first two horizontal actuators are inverted=true/right=true; the
        # rear two are inverted=false/right=false. Columns are
        # FrontRight, FrontLeft, BackRight, BackLeft; rows are Fx, Fy, Nz.
        # This is A^T(AA^T)^-1 for that matrix at force_per_pwm=0.8.
        # Keep it explicit so the safety node has no NumPy/runtime dependency.
        if abs(self.force_per_pwm - 0.8) > 1e-9:
            raise rospy.ROSInitException(
                "force_per_normalized_pwm must remain 0.8 until allocator calibration")
        self.pinv = (
            (-0.441941738,  0.453605890,  1.832844575),
            (-0.441941738, -0.453605890, -1.832844575),
            (0.441941738,  0.430277587, -1.832844575),
            (0.441941738, -0.430277587,  1.832844575),
        )
        self.pub = rospy.Publisher("/bluerov2/setpoint/pwm", Float64MultiArray, queue_size=1)
        rospy.Subscriber("/controller/generalized_force", WrenchStamped, self.cb, queue_size=1)
        self.timer = rospy.Timer(rospy.Duration(0.05), self.update)
        rospy.on_shutdown(self.shutdown)

    def cb(self, msg):
        self.last_wrench = msg

    def update(self, _event):
        out = [0.0] * 8
        if self.enabled and self.last_wrench is not None:
            age = (rospy.Time.now() - self.last_wrench.header.stamp).to_sec()
            if 0.0 <= age <= self.timeout:
                w = self.last_wrench.wrench
                wrench = (w.force.x, w.force.y, w.torque.z)
                for i in range(4):
                    out[i] = max(-1.0, min(1.0, sum(self.pinv[i][j] * wrench[j] for j in range(3))))
                vertical_pwm = max(-1.0, min(1.0, w.force.z / (4.0 * self.force_per_pwm)))
                out[4:8] = [vertical_pwm] * 4
        self.pub.publish(Float64MultiArray(data=out))

    def shutdown(self):
        self.enabled = False
        self.pub.publish(Float64MultiArray(data=[0.0] * 8))


if __name__ == "__main__":
    rospy.init_node("blue8_thruster_allocator")
    Blue8Allocator()
    rospy.spin()
