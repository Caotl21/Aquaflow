#!/usr/bin/env python3
"""
临时修复：在 HOFA 逆变换后反转 x 方向的力

这是一个诊断性修复，用于测试问题是否在 MPC 求解阶段
"""

import rospy
from geometry_msgs.msg import WrenchStamped

def callback(msg):
    # 反转 x 方向的力
    corrected = WrenchStamped()
    corrected.header = msg.header
    corrected.wrench.force.x = -msg.wrench.force.x  # 反转！
    corrected.wrench.force.y = msg.wrench.force.y
    corrected.wrench.force.z = msg.wrench.force.z
    corrected.wrench.torque.x = msg.wrench.torque.x
    corrected.wrench.torque.y = msg.wrench.torque.y
    corrected.wrench.torque.z = msg.wrench.torque.z

    pub.publish(corrected)

    rospy.loginfo_throttle(1.0, "Force inversion: Fx %.2f -> %.2f",
                          msg.wrench.force.x, corrected.wrench.force.x)

if __name__ == "__main__":
    rospy.init_node("force_inverter_test")

    # 重新映射：
    # MPC 输出到 /controller/generalized_force_original
    # 这个节点读取它，反转 Fx，发布到 /controller/generalized_force

    sub = rospy.Subscriber("/controller/generalized_force_original",
                          WrenchStamped, callback, queue_size=1)
    pub = rospy.Publisher("/controller/generalized_force",
                         WrenchStamped, queue_size=1)

    rospy.logwarn("=" * 70)
    rospy.logwarn("力反转测试节点已启动")
    rospy.logwarn("这会将 MPC 输出的 Fx 反转")
    rospy.logwarn("如果机器人开始正常前进，说明 MPC 求解方向错误")
    rospy.logwarn("=" * 70)

    rospy.spin()
