#!/usr/bin/env python3
"""
快速测试：反转参考速度方向
如果这个修复有效，说明参考轨迹处理器生成的速度方向错误
"""

import rospy
from hofa_mpc_ros.msg import TrajectoryPoint
from geometry_msgs.msg import Twist

def callback(msg):
    # 创建修正后的消息
    corrected = msg
    # 反转速度方向
    corrected.twist.linear.x = -msg.twist.linear.x
    corrected.twist.linear.y = -msg.twist.linear.y
    corrected.twist.angular.z = -msg.twist.angular.z
    # 反转加速度方向
    corrected.accel.linear.x = -msg.accel.linear.x
    corrected.accel.linear.y = -msg.accel.linear.y
    corrected.accel.angular.z = -msg.accel.angular.z

    pub.publish(corrected)
    rospy.loginfo_throttle(1.0, "反转参考速度: v=(%.2f, %.2f) -> (%.2f, %.2f)",
                          msg.twist.linear.x, msg.twist.linear.y,
                          corrected.twist.linear.x, corrected.twist.linear.y)

if __name__ == "__main__":
    rospy.init_node("reference_velocity_inverter")

    # 订阅原始参考轨迹
    rospy.Subscriber("/controller/reference_trajectory_original",
                     TrajectoryPoint, callback, queue_size=1)

    # 发布修正后的参考轨迹
    pub = rospy.Publisher("/controller/reference_trajectory",
                         TrajectoryPoint, queue_size=1)

    rospy.loginfo("参考速度反转器已启动")
    rospy.loginfo("这是一个测试节点，用于诊断速度方向问题")
    rospy.spin()
