import unittest
import numpy as np

from road_identification import estimate_road_length

class TestRoadWidthDetection(unittest.TestCase):
    threshold = 1.0
    adjust_npixel = 0
    tolerance = 0.1

    def test_roadwidth_detection(self):
        pixels = np.array([
            [0, 9, 9, 9, 9],
            [9, 9, 9, 9, 0]
            ])
        offsets, road_pixels = estimate_road_length(pixels, self.threshold, self.adjust_npixel)
        self.assertEqual(offsets, [(1, 5), (0, 4)])
        np.testing.assert_array_equal(
                road_pixels,
                np.array([
                    [0, 1, 1, 1, 1],
                    [1, 1, 1, 1, 0]
                    ])
                )
