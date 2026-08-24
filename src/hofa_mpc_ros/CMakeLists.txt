cmake_minimum_required(VERSION 3.0.2)
project(hofa_mpc_ros)

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  geometry_msgs
  nav_msgs
  message_generation
)

catkin_python_setup()

add_message_files(
  FILES
  TrajectoryPoint.msg
  TrajectoryPointWindow.msg
  ControllerStatus.msg
  ActuatorStatus.msg
)

generate_messages(
  DEPENDENCIES
  std_msgs
  geometry_msgs
  nav_msgs
)

catkin_package(
  CATKIN_DEPENDS rospy std_msgs geometry_msgs nav_msgs message_runtime
)

catkin_install_python(PROGRAMS
  scripts/hofa_mpc_controller_node.py
  scripts/trajectory_server_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY config launch
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
