cmake_minimum_required(VERSION 3.0.2)
project(hofa_policy)

find_package(catkin REQUIRED COMPONENTS
  rospy
  geometry_msgs
  nav_msgs
  sensor_msgs
  std_msgs
  hofa_mpc_ros
)

catkin_python_setup()

catkin_package(
  CATKIN_DEPENDS rospy geometry_msgs nav_msgs sensor_msgs std_msgs hofa_mpc_ros
)

catkin_install_python(PROGRAMS
  scripts/hofa_policy_trajectory_publisher.py
  scripts/hofa_policy_shadow_node.py
  scripts/hofa_policy_controller_node.py
  scripts/hofa_policy_perturbation_relay.py
  scripts/run_policy_comparison.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(PROGRAMS
  scripts/hofa_policy_shadow_node.sh
  scripts/hofa_policy_controller_node.sh
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY config
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
