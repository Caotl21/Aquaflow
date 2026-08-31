#!/usr/bin/env python3
"""Simple test command publisher for thruster allocation testing."""
import rospy
from geometry_msgs.msg import WrenchStamped


def main():
    rospy.init_node("thruster_test_publisher")
    pub = rospy.Publisher("/controller/generalized_force", WrenchStamped, queue_size=1)

    # Get test mode from parameter
    mode = rospy.get_param("~mode", "forward")

    rate = rospy.Rate(10)  # 10 Hz
    rospy.sleep(1.0)  # Wait for connections

    rospy.loginfo("Publishing %s test command...", mode)

    while not rospy.is_shutdown():
        msg = WrenchStamped()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = "base_link"

        if mode == "forward":
            msg.wrench.force.x = 10.0
        elif mode == "lateral":
            msg.wrench.force.y = 10.0
        elif mode == "yaw":
            msg.wrench.torque.z = 5.0
        elif mode == "backward":
            msg.wrench.force.x = -10.0
        else:
            rospy.logwarn_once("Unknown mode: %s, using forward", mode)
            msg.wrench.force.x = 10.0

        pub.publish(msg)
        rate.sleep()


if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass
