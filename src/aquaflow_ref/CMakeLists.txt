cmake_minimum_required(VERSION 3.0.2)
project(aquaflow_ref)

find_package(catkin REQUIRED COMPONENTS geometry_msgs nav_msgs rospy std_msgs visualization_msgs)
catkin_package()

catkin_install_python(PROGRAMS
  scripts/privileged_teacher.py
  scripts/privileged_map_visualizer.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(FILES scripts/teacher_core.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY config launch
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
