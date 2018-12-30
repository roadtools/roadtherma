import unittest
import numpy as np

from pave_analyser.road_identification import estimate_road_length


class DummyTemperatures:
    def __init__(self, array):
        self.values = array


class DummyData:
    def __init__(self, array):
        self.temperatures = DummyTemperatures(array)


class TestRoadWidthDetection(unittest.TestCase):
    threshold = 1.0
    adjust_npixel = 0
    tolerance = 0.1

    def test_roadwidth_detection(self):
        pixels = np.array([
            [0, 9, 9, 9, 9],
            [9, 9, 9, 9, 0]
            ])

        data = DummyData(pixels)
        estimate_road_length(data, self.threshold, self.adjust_npixel)
        self.assertEqual(data.offsets, [(1, 5), (0, 4)])
        np.testing.assert_array_equal(
                data.road_pixels,
                np.array([
                    [0, 1, 1, 1, 1],
                    [1, 1, 1, 1, 0]
                    ])
                )
