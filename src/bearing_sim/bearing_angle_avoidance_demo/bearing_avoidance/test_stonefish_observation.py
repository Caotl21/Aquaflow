#!/usr/bin/env python3
"""Pure-Python checks for the Stonefish image observation adapter."""
import math

import numpy as np

from stonefish_observation import (CameraModel, ColorConfig, SegmentationConfig,
                                   extract_color_components, extract_observations,
                                   image_to_rgb, image_to_uint16,
                                   truth_angle_observations, truth_angle_sensor)


def main():
    camera = CameraModel.from_values(640, 480, 458.366, cx=320.0, cy=240.0)
    labels = np.zeros((480, 640), dtype=np.uint16)
    labels[180:360, 300:340] = 11
    result = extract_observations(labels, camera)
    assert result.accepted_labels == (11,)
    beta, theta = result.detections["11"]
    assert abs(beta) < math.radians(2.0)
    assert theta > 0.0

    labels[:, :400] = 22  # scene-sized label must not become an obstacle.
    labels[180:360, 300:340] = 11
    result = extract_observations(labels, camera)
    assert "22" not in result.detections
    assert "11" in result.detections

    labels = np.zeros((480, 640), dtype=np.uint16)
    labels[180:360, 300:340] = 11
    result = extract_observations(
        labels, camera, SegmentationConfig(min_height=400), allowed_labels=[11])
    assert result.detections == {}
    assert len(result.partial_sectors) == 1

    encoded = labels.astype("<u2").tobytes()
    decoded = image_to_uint16(encoded, 480, 640, 640 * 2, False)
    assert np.array_equal(labels, decoded)
    rgb = np.full((480, 640, 3), 180, dtype=np.uint8)
    rgb[180:420, 300:335] = [18, 20, 22]
    components, mask = extract_color_components(
        rgb, camera, ColorConfig(dark_max_value=65, min_pixels=40, min_height=12))
    assert len(components) == 1 and components[0].complete
    assert mask.dtype == np.uint8 and mask.max() == 255
    rgb[80:130, :] = [18, 20, 22]  # horizontal water-surface-like strip
    components, _ = extract_color_components(
        rgb, camera, ColorConfig(dark_max_value=65, min_pixels=40, min_height=12))
    assert len(components) == 1 and components[0].bottom >= 400
    assert np.array_equal(rgb, image_to_rgb(rgb.tobytes(), 480, 640, 640 * 3, "rgb8"))

    # Stonefish-like frame: lit blue background, dark surface region above
    # the horizon, a near pillar wider than the old 20% gate whose top merges
    # with the surface, and blue-tinted distant pillars.
    rgb = np.empty((480, 640, 3), dtype=np.uint8)
    rgb[:] = [26, 122, 153]
    rgb[0:105, 0:271] = [10, 25, 30]
    rgb[60:480, 150:450] = [10, 27, 36]      # near pillar, 300 px = 47% width
    rgb[150:450, 480:500] = [11, 43, 55]     # distant, spread 44
    rgb[200:420, 520:532] = [16, 58, 73]     # farther, spread 57
    components, _ = extract_color_components(rgb, camera)
    found = sorted((c.left, c.right, c.complete) for c in components)
    assert found == [(150, 449, True), (480, 499, True), (520, 531, True)], found
    rgb[60:480, 0:150] = [10, 27, 36]        # pillar now touches the image side
    components, _ = extract_color_components(rgb, camera)
    near = [c for c in components if c.left == 0]
    assert len(near) == 1 and not near[0].complete and near[0].right == 449
    rgb[:] = [26, 122, 153]
    rgb[0:105, :] = [10, 25, 30]             # surface alone is not a pillar
    rgb[400:480, :] = [10, 25, 30]           # nor is a dark floor strip
    components, _ = extract_color_components(rgb, camera)
    assert components == []

    # Overlapping pillars: the nearer one is darker (less attenuation) or
    # meets the floor lower; it stays complete, the occluded one is partial.
    rgb[:] = [26, 122, 153]
    rgb[100:480, 300:361] = [8, 40, 50]      # far pillar, partly hidden
    rgb[0:480, 240:321] = [8, 25, 33]        # near, darker pillar in front
    found = sorted((c.left, c.right, c.complete) for c in extract_color_components(rgb, camera)[0])
    assert found == [(240, 320, True), (321, 360, False)], found
    rgb[:] = [26, 122, 153]
    rgb[100:400, 230:271] = [8, 40, 50]      # far: same colour, floor row 399
    rgb[0:480, 200:241] = [8, 40, 50]        # near: runs off the image bottom
    found = sorted((c.left, c.right, c.complete) for c in extract_color_components(rgb, camera)[0])
    assert found == [(200, 240, True), (241, 270, False)], found
    # rgb_dense_s103: lighter silhouette columns (shading/anti-aliasing) used
    # to be cut off as a 2-3 px occluded sliver, i.e. a partial sector.
    rgb[:] = [26, 122, 153]
    rgb[0:480, 240:321] = [8, 25, 33]
    rgb[0:480, 321:324] = [8, 40, 50]
    found = sorted((c.left, c.right, c.complete) for c in extract_color_components(rgb, camera)[0])
    assert found == [(240, 323, True)], found

    truth = truth_angle_observations(
        np.array([0.0, 0.0, 0.0]), [[2.0, 1.0], [-2.0, 0.0]],
        [1.0, 0.5], [12, 14], fov_deg=140.0, max_range=5.0)
    assert set(truth) == {"12"}
    assert np.isclose(truth["12"][0], math.atan2(1.0, 2.0))
    assert np.isclose(truth["12"][1], 2.0 * math.asin(1.0 / (2.0 * math.sqrt(5.0))))
    assert truth["12"].shape == (2,)

    # Without occlusion/noise the sensor matches the exact helper; with
    # occlusion the pillar hidden behind 12 is withheld, and a half-hidden
    # pillar becomes a partial sector instead of a complete detection.
    pillars, diameters, labels = [[2.0, 0.0], [4.0, 0.0], [4.0, 0.45]], [0.6, 0.6, 0.6], [12, 14, 16]
    pose = np.array([0.0, 0.0, 0.0])
    exact = truth_angle_observations(pose, pillars, diameters, labels, 140.0, 13.0)
    detections, partial = truth_angle_sensor(pose, pillars, diameters, labels, 140.0, 13.0)
    assert set(detections) == set(exact) == {"12", "14", "16"} and not partial
    assert all(np.allclose(detections[k], exact[k]) for k in exact)
    detections, partial = truth_angle_sensor(pose, pillars, diameters, labels, 140.0, 13.0,
                                             occlusion=True)
    assert set(detections) == {"12"}
    assert len(partial) == 1 and partial[0][1] > exact["16"][0]
    assert partial[0][0] >= exact["12"][0] + exact["12"][1] / 2.0 - 1e-3
    rng = np.random.default_rng(3)
    noisy = [truth_angle_sensor(pose, pillars[:1], diameters[:1], labels[:1], 140.0, 13.0,
                                noise_deg=(0.6, 0.35), rng=rng)[0]["12"] for _ in range(400)]
    spread = np.degrees(np.std(np.asarray(noisy) - exact["12"], axis=0))
    assert np.allclose(spread, [0.6, 0.35], rtol=0.2)
    print("PASS: camera projection, scene-label filtering, partial sectors, ROS 16UC1 decoding")


if __name__ == "__main__":
    main()

