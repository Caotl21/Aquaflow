#!/usr/bin/env python3
"""Print one Stonefish segmentation frame's label bounding boxes.

Use this before P1 closed-loop control to identify which object IDs are the
black pillars. The output is diagnostic only and is not used by the planner.
"""
import math
from pathlib import Path
import sys

import numpy as np
import rospy
from sensor_msgs.msg import CameraInfo, Image

# Import the ROS-free adapter directly. Importing the main node would resolve
# through catkin's generated relay and make this diagnostic script fragile.
adapter_root = Path(__file__).resolve().parents[2] / "bearing_sim" / "bearing_angle_avoidance_demo" / "bearing_avoidance"
if str(adapter_root) not in sys.path:
    sys.path.insert(0, str(adapter_root))
from stonefish_observation import CameraModel, _label_bbox, image_to_uint16


class Inspector:
    def __init__(self):
        self.image_topic = rospy.get_param("~image_topic", "/bricsbot/bearing_segmentation/image_raw")
        self.info_topic = rospy.get_param("~camera_info_topic", "/bricsbot/bearing_segmentation/camera_info")
        self.camera = None
        self.done = False
        rospy.Subscriber(self.info_topic, CameraInfo, self.info_cb, queue_size=1)
        rospy.Subscriber(self.image_topic, Image, self.image_cb, queue_size=1)

    def info_cb(self, msg):
        fx = msg.K[0]
        fy = msg.K[4] or fx
        self.camera = CameraModel.from_values(msg.width, msg.height, fx, fy, msg.K[2], msg.K[5])

    def image_cb(self, msg):
        if self.done or self.camera is None:
            return
        labels = image_to_uint16(msg.data, msg.height, msg.width, msg.step, msg.is_bigendian)
        values, counts = np.unique(labels, return_counts=True)
        print("label,count,left,right,top,bottom,area,bearing_left_deg,bearing_right_deg")
        for label, count in sorted(zip(values.tolist(), counts.tolist()), key=lambda item: item[1], reverse=True):
            label = int(label)
            if label == 0:
                continue
            bbox = _label_bbox(labels, label)
            left, right, top, bottom, area = bbox
            beta_left = math.degrees(math.atan2(left - self.camera.cx, self.camera.fx))
            beta_right = math.degrees(math.atan2(right + 1 - self.camera.cx, self.camera.fx))
            print("%d,%d,%d,%d,%d,%d,%d,%.3f,%.3f" %
                  (label, count, left, right, top, bottom, area, beta_left, beta_right))
        self.done = True
        rospy.signal_shutdown("one segmentation frame inspected")


if __name__ == "__main__":
    rospy.init_node("inspect_segmentation_labels")
    Inspector()
    rospy.spin()

