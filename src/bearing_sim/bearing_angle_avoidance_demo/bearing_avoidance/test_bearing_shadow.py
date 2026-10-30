"""Run with python -m unittest discover -s <this directory> -p test_bearing_shadow.py."""
import math
import unittest
from dataclasses import replace

import numpy as np

from bearing_shadow import ComparisonSummary, camera_pose, compare_truth_and_segmentation, match_odometry
from stonefish_observation import CameraModel, SegmentationConfig


class TimingTests(unittest.TestCase):
    def test_exact_and_yaw_wrap(self):
        samples = [(10., 0., 0., math.radians(179)), (10.02, 2., 0., math.radians(-179))]
        exact = match_odometry(10., samples)
        np.testing.assert_allclose(exact.pose, samples[0][1:])
        pair = match_odometry(10.01, samples)
        self.assertEqual(pair.mode, "interpolated")
        self.assertAlmostEqual(pair.pose[0], 1.)
        self.assertAlmostEqual(abs(pair.pose[2]), math.pi)
        self.assertAlmostEqual(pair.bracket_span_s, .02)
        self.assertAlmostEqual(pair.nearest_skew_s, .01)

    def test_no_interpolation_across_missing_odometry(self):
        pair = match_odometry(10.01, [(10., 0, 0, 0), (12., 2, 0, 0)])
        self.assertIsNone(pair.pose)
        self.assertEqual(pair.mode, "odometry_gap")

    def test_nearest_is_bounded_and_reported(self):
        samples = [(10., 1., 2., .3)]
        self.assertEqual(match_odometry(10.02, samples).mode, "nearest")
        self.assertAlmostEqual(match_odometry(10.02, samples).nearest_skew_s, .02)
        self.assertIsNone(match_odometry(10.2, samples).pose)
        self.assertIsNone(match_odometry(9.8, samples).pose)
        self.assertIsNone(match_odometry(0., samples).pose)
        self.assertIsNone(match_odometry(10., []).pose)

    def test_unsorted_samples_are_not_silently_used(self):
        with self.assertRaises(ValueError):
            match_odometry(10., [(11., 0, 0, 0), (9., 0, 0, 0)])

    def test_camera_lever_arm_rotates_with_heading(self):
        np.testing.assert_allclose(camera_pose([1., 2., math.pi/2], [.2, 0.]),
                                   [1., 2.2, math.pi/2])


class ComparisonTests(unittest.TestCase):
    def setUp(self):
        self.camera = CameraModel.from_values(640, 480, 320.)
        self.mask = np.zeros((480, 640), np.uint16)
        self.mask[100:400, 300:340] = 12
        self.diameter = 4. * math.sin(math.atan2(20., 320.))
        self.config = SegmentationConfig()

    def compare(self, centers=None, diameters=None, labels=None, **kwargs):
        return compare_truth_and_segmentation(self.mask, self.camera,
            np.zeros(3), centers or [[2., 0.]], diameters or [self.diameter],
            labels or [12], config=self.config, allowed_labels=kwargs.pop("allowed_labels", [12, 14]),
            **kwargs)

    def test_exact_geometry_and_no_mutation(self):
        old = self.mask.copy()
        row = self.compare()[0]
        self.assertEqual(row["segmentation_status"], "accepted_complete")
        self.assertEqual(row["predicted_occlusion"], "unoccluded")
        self.assertAlmostEqual(row["bearing_error_rad"], 0.)
        self.assertAlmostEqual(row["width_error_rad"], 0.)
        np.testing.assert_array_equal(old, self.mask)

    def test_fixed_pixel_shift_has_signed_error(self):
        self.mask[:] = 0
        self.mask[100:400, 310:350] = 12
        row = self.compare()[0]
        self.assertGreater(row["bearing_error_rad"], 0.)
        self.assertLess(row["width_error_rad"], 0.)

    def test_hidden_pillar_keeps_missing_separate_from_zero_error(self):
        rows = self.compare([[2., 0.], [4., 0.]], [self.diameter, self.diameter], [12, 14])
        hidden = rows[1]
        self.assertEqual(hidden["truth_available"], 1)
        self.assertEqual(hidden["predicted_occlusion"], "occluded")
        self.assertEqual(hidden["segmentation_status"], "missing")
        self.assertIsNone(hidden["width_error_rad"])

    def test_accepted_bbox_can_still_be_occluded(self):
        self.mask[100:400, 340:350] = 14
        rows = self.compare([[2., 0.], [4., .25]], [self.diameter, .5], [12, 14])
        self.assertEqual(rows[1]["segmentation_status"], "accepted_complete")
        self.assertEqual(rows[1]["predicted_occlusion"], "partial_occlusion")
        self.assertLess(rows[1]["width_error_rad"], 0.)

    def test_partial_and_rejection_reasons(self):
        self.mask[:] = 0
        self.mask[:, 0:30] = 12
        row = self.compare()[0]
        self.assertEqual(row["segmentation_status"], "partial")
        self.assertIn("image_edge", row["rejection_reason"])
        self.assertIsNone(row["width_error_rad"])
        self.assertIsNotNone(row["bbox_width_error_rad"])
        self.mask[:] = 0
        self.mask[200, 320] = 12
        row = self.compare()[0]
        self.assertEqual(row["segmentation_status"], "rejected")
        self.assertIn("low_pixels", row["rejection_reason"])

    def test_filtered_labels_and_candidate_limit(self):
        row = self.compare(allowed_labels=[14])[0]
        self.assertEqual(row["rejection_reason"], "label_filtered")
        self.mask[100:400, 400:460] = 14
        self.config = replace(self.config, max_labels=1)
        row = self.compare()[0]
        self.assertEqual(row["segmentation_status"], "rejected")
        self.assertEqual(row["rejection_reason"], "candidate_limit")

    def test_near_wide_object_is_partial_not_absent(self):
        self.mask[:, 200:440] = 12
        row = self.compare()[0]
        self.assertEqual(row["segmentation_status"], "partial")
        self.assertIn("too_wide", row["rejection_reason"])

    def test_summary_excludes_missing_and_separates_occluded_errors(self):
        rows = self.compare([[2., 0.], [4., 0.]], [self.diameter, self.diameter], [12, 14])
        summary = ComparisonSummary()
        summary.add_frame("compared", rows)
        summary.add_frame("unpaired")
        report = summary.result()
        self.assertEqual(report["frame_status"], dict(compared=1, unpaired=1))
        self.assertEqual(report["counts"]["oracle_available_missing"], 1)
        self.assertEqual(report["errors"]["accepted_width"]["count"], 1)
        self.assertEqual(report["errors"]["unoccluded_full_fov_width"]["count"], 1)
        self.assertEqual(report["per_label"]["14"]["errors"], {})

    def test_invalid_geometry_labels_rejected(self):
        with self.assertRaises(ValueError):
            self.compare([[2., 0.], [4., 0.]], [self.diameter]*2, [12, 12])


if __name__ == "__main__":
    unittest.main()
