#!/usr/bin/env bash
# Source this file in every new terminal before using Stonefish/ROS:
#   source /home/bricsbot/catkin_ws/scripts/setup_stonefish_ros.sh

unset ROS_IP
export ROS_HOSTNAME=localhost
export ROS_MASTER_URI=http://localhost:11311

_aquaflow_stonefish_lib=/home/bricsbot/simulators/stonefish-1.6/install/lib
case ":${LD_LIBRARY_PATH:-}:" in
  *":${_aquaflow_stonefish_lib}:"*) ;;
  *) export LD_LIBRARY_PATH="${_aquaflow_stonefish_lib}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" ;;
esac
unset _aquaflow_stonefish_lib

export __NV_PRIME_RENDER_OFFLOAD=1
export __GLX_VENDOR_LIBRARY_NAME=nvidia

if [[ -f /opt/ros/noetic/setup.bash ]]; then
  source /opt/ros/noetic/setup.bash
else
  echo "AquaFlow setup error: /opt/ros/noetic/setup.bash not found." >&2
  return 1 2>/dev/null || exit 1
fi

if [[ -f /home/bricsbot/catkin_ws/devel/setup.bash ]]; then
  source /home/bricsbot/catkin_ws/devel/setup.bash
else
  echo "AquaFlow setup error: workspace devel/setup.bash not found; run catkin_make first." >&2
  return 1 2>/dev/null || exit 1
fi

echo "Stonefish/ROS environment ready."
echo "  ROS_MASTER_URI=${ROS_MASTER_URI}"
echo "  ROS_HOSTNAME=${ROS_HOSTNAME}"
echo "  workspace=/home/bricsbot/catkin_ws"
