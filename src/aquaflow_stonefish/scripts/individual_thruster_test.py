#!/usr/bin/env python3
"""Individual thruster test - directly publish PWM commands."""
import rospy
from std_msgs.msg import Float64MultiArray


def main():
    rospy.init_node("individual_thruster_test")
    pub = rospy.Publisher("/bricsbot/setpoint/pwm", Float64MultiArray, queue_size=1)

    rospy.sleep(2.0)

    tests = [
        ("T1 only (+0.2)", [0.2, 0.0, 0.0, 0.0, 0.0, 0.0]),
        ("T2 only (+0.2)", [0.0, 0.2, 0.0, 0.0, 0.0, 0.0]),
        ("T3 only (+0.2)", [0.0, 0.0, 0.2, 0.0, 0.0, 0.0]),
        ("T4 only (+0.2)", [0.0, 0.0, 0.0, 0.2, 0.0, 0.0]),
        ("T1+T2 forward", [0.2, 0.2, 0.0, 0.0, 0.0, 0.0]),
        ("T3+T4 backward", [0.0, 0.0, -0.2, -0.2, 0.0, 0.0]),
        ("All forward", [0.2, 0.2, -0.2, -0.2, 0.0, 0.0]),
    ]

    rate = rospy.Rate(0.2)  # 5 seconds per test

    for name, pwm in tests:
        rospy.loginfo("Testing: %s -> PWM: %s", name, pwm)
        msg = Float64MultiArray(data=pwm)

        # Publish for 5 seconds
        for _ in range(10):
            pub.publish(msg)
            rospy.sleep(0.5)

        # Stop
        pub.publish(Float64MultiArray(data=[0.0]*6))
        rospy.loginfo("Stop - next test in 3 seconds...")
        rospy.sleep(3.0)

    rospy.loginfo("All tests complete")


if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass
