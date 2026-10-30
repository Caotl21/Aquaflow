#!/usr/bin/env python3
"""Save one RGB frame and its black-pillar mask, and explain every rejection.

Diagnostic only: use this to tune the RGB thresholds when the bearing PF node
sees no pillars. Writes ``rgb.png``, ``dark_mask.png`` and ``report.txt`` to
``~out_dir`` and prints the same report.
"""
from pathlib import Path
import math
import sys

import numpy as np
import rospy
from sensor_msgs.msg import CameraInfo, Image

adapter_root = Path(__file__).resolve().parents[2] / "bearing_sim" / "bearing_angle_avoidance_demo" / "bearing_avoidance"
if str(adapter_root) not in sys.path:
    sys.path.insert(0, str(adapter_root))
import cv2
from stonefish_observation import CameraModel, ColorConfig, extract_color_components, image_to_rgb


class Inspector:
    def __init__(self):
        self.image_topic = rospy.get_param("~image_topic", "/bricsbot/camera_front/image_color")
        self.info_topic = rospy.get_param("~camera_info_topic", "/bricsbot/camera_front/camera_info")
        self.out_dir = Path(rospy.get_param("~out_dir", "/tmp/bearing_rgb_inspect"))
        self.config = ColorConfig(
            dark_max_value=int(rospy.get_param("~rgb_dark_max_value", 80)),
            dark_channel_spread=int(rospy.get_param("~rgb_dark_channel_spread", 60)),
            min_pixels=int(rospy.get_param("~rgb_min_pixels", 40)),
            min_height=int(rospy.get_param("~rgb_min_height_px", 12)),
            min_width_px=int(rospy.get_param("~rgb_min_width_px", 2)),
            band_half_height_px=int(rospy.get_param("~rgb_band_half_height_px", 24)),
            band_fill=float(rospy.get_param("~rgb_band_fill", 0.6)),
            partial_margin_px=int(rospy.get_param("~partial_margin_px", 2)))
        self.camera = None
        self.done = False
        self.image_count = 0
        rospy.Subscriber(self.info_topic, CameraInfo, self.info_cb, queue_size=1)
        rospy.Subscriber(self.image_topic, Image, self.image_cb, queue_size=1)
        rospy.loginfo("waiting for %s and %s; output -> %s",
                      self.image_topic, self.info_topic, self.out_dir)
        rospy.Timer(rospy.Duration(3.0), self.waiting_cb)

    def waiting_cb(self, _event):
        if not self.done:
            rospy.logwarn("still waiting: camera_info=%s, images received=%d (is the sim running "
                          "and unpaused? check `rostopic hz %s`)",
                          "ok" if self.camera is not None else "MISSING",
                          self.image_count, self.image_topic)

    def info_cb(self, msg):
        fx = msg.K[0]
        self.camera = CameraModel.from_values(msg.width, msg.height, fx, msg.K[4] or fx, msg.K[2], msg.K[5])

    def image_cb(self, msg):
        self.image_count += 1
        if self.done or self.camera is None:
            return
        self.done = True
        rgb = image_to_rgb(msg.data, msg.height, msg.width, msg.step, msg.encoding, msg.is_bigendian)
        rejected = []
        components, mask = extract_color_components(rgb, self.camera, self.config, rejected)
        values = rgb.astype(np.int32)
        brightness = values.max(axis=2)
        spread = brightness - values.min(axis=2)
        lines = ["encoding=%s size=%dx%d config=%s" % (msg.encoding, msg.width, msg.height, self.config),
                 "dark pixels: %d (%.1f%%)" % (int((mask > 0).sum()), 100.0 * (mask > 0).mean())]
        # Pixel statistics per column band help tell "pillar too bright/tinted"
        # apart from "pillar merged with background".
        for name, band in (("centre band (cols 280-360)", slice(280, 360)),
                           ("whole image", slice(None))):
            b, s = brightness[:, band], spread[:, band]
            lines.append("%s: max(RGB) p5/p50/p95=%s  spread p5/p50/p95=%s  mean RGB=%s"
                         % (name, np.percentile(b, [5, 50, 95]).astype(int).tolist(),
                            np.percentile(s, [5, 50, 95]).astype(int).tolist(),
                            rgb[:, band].reshape(-1, 3).mean(axis=0).round(1).tolist()))
        lines.append("accepted components: %d" % len(components))
        for c in components:
            lines.append("  cols %d-%d rows %d-%d area=%d beta=%.1fdeg width=%.1fdeg complete=%s"
                         % (c.left, c.right, c.top, c.bottom, c.area,
                            math.degrees(c.beta), math.degrees(c.theta), c.complete))
        rejected.sort(key=lambda item: item[4], reverse=True)
        lines.append("rejected components (largest 15 of %d):" % len(rejected))
        for left, top, width, height, area, reason in rejected[:15]:
            lines.append("  cols %d-%d rows %d-%d area=%d -> %s"
                         % (left, left + width - 1, top, top + height - 1, area, reason))
        report = "\n".join(lines)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(self.out_dir / "rgb.png"), np.ascontiguousarray(rgb[..., ::-1]))
        cv2.imwrite(str(self.out_dir / "dark_mask.png"), mask)
        (self.out_dir / "report.txt").write_text(report + "\n")
        print(report)
        print("saved to %s" % self.out_dir)
        rospy.signal_shutdown("done")


if __name__ == "__main__":
    rospy.init_node("inspect_rgb_pillars", anonymous=True)
    Inspector()
    rospy.spin()
