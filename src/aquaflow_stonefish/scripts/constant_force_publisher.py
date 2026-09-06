#!/usr/bin/env python3
"""Publish a fresh-timestamp constant body-frame wrench for calibration.

Unlike ``rostopic pub -r``, this node rebuilds the WrenchStamped header on
every cycle, so brics6_thruster_allocator's watchdog never sees a stale stamp.
"""
import rospy
from geometry_msgs.msg import WrenchStamped


def main():
    rospy.init_node("constant_force_publisher")
    rate_hz = max(1.0, float(rospy.get_param("~rate", 20.0)))
    force_x = float(rospy.get_param("~force_x", 0.0))
    force_y = float(rospy.get_param("~force_y", 0.0))
    force_z = float(rospy.get_param("~force_z", 0.0))
    torque_z = float(rospy.get_param("~torque_z", 0.0))
    topic = rospy.get_param("~topic", "/controller/generalized_force")
    pub = rospy.Publisher(topic, WrenchStamped, queue_size=1)
    rate = rospy.Rate(rate_hz)

    rospy.loginfo("Constant wrench publisher: Fx=%.3f Fy=%.3f Fz=%.3f Nz=%.3f N/Nm at %.1f Hz",
                  force_x, force_y, force_z, torque_z, rate_hz)
    while not rospy.is_shutdown():
        msg = WrenchStamped()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = "base_link"
        msg.wrench.force.x = force_x
        msg.wrench.force.y = force_y
        msg.wrench.force.z = force_z
        msg.wrench.torque.z = torque_z
        pub.publish(msg)
        rate.sleep()


if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass
