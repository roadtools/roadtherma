import unittest
import numpy as np
import pandas as pd

from roadtherma.road_identification import estimate_road_length, detect_paving_lanes


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


class TestLaneSelector(unittest.TestCase):
    threshold = 9.5

    def _execute_test(self, df, values_verify, columns_verify, nlanes, selector):
        data = DummyData(None)
        data.df = df
        number_of_lanes = detect_paving_lanes(data, self.threshold, select=selector)
        values_after = data.df.values
        self.assertEqual(number_of_lanes, nlanes)
        self.assertEqual(
                columns_verify,
                list(data.df.columns)
                )
        np.testing.assert_array_equal(
                values_after,
                values_verify
                )


    def test_single_lane_detection(self):
        df = pd.DataFrame.from_dict({
            'T1':[9, 9, 9, 9],
            'T2':[10,10,10,10],
            'T3':[10,10,10,10],
            'T4':[10,10,10,10],
            'T5':[9, 9, 9, 9],
            })
        columns_verify = ['T1', 'T2', 'T3', 'T4', 'T5']
        values_verify = np.array([
            [9, 9, 9, 9],
            [10,10,10,10],
            [10,10,10,10],
            [10,10,10,10],
            [9, 9, 9, 9],
            ]).T
        nlanes = 1
        self._execute_test(df, values_verify, columns_verify, nlanes, 'warmest')

    def test_double_lane_detection(self):
        df = pd.DataFrame.from_dict({
            'T1':[9, 9, 9, 9],
            'T2':[10,10,10,11], #<-- warmest! :)
            'T3':[9, 9,  9, 9],
            'T4':[10,10,10,10],
            'T5':[9, 9, 9, 9],
            })
        columns_verify = ['T1', 'T2', 'T3']
        values_verify = np.array([
            [9, 9, 9, 9],
            [10,10,10,11],
            [9, 9,  9, 9],
            ]).T
        nlanes = 2
        self._execute_test(df, values_verify, columns_verify, nlanes, 'warmest')

    def test_coldest_selector(self):
        df = pd.DataFrame.from_dict({
            'T1':[9, 9, 9, 9],
            'T2':[10,10,10,11],
            'T3':[9, 9,  9, 9],
            'T4':[10,10,10,10], #<-- coldest! :)
            'T5':[9, 9, 9, 9],
            })
        columns_verify = ['T3', 'T4', 'T5']
        values_verify = np.array([
            [9, 9, 9, 9],
            [10,10,10,10],
            [9, 9, 9, 9],
            ]).T
        nlanes = 2
        self._execute_test(df, values_verify, columns_verify, nlanes, 'coldest')

    def test_no_cold_edge_selector(self):
        df = pd.DataFrame.from_dict({
            'T2':[10,10,10,11], #<-- warmest! :)
            'T3':[9, 9,  9, 9],
            'T4':[10,10,10,10],
            'T5':[9, 9, 9, 9],
            })
        columns_verify = ['T2', 'T3']
        values_verify = np.array([
            [10,10,10,11],
            [9, 9,  9, 9],
            ]).T
        nlanes = 2
        self._execute_test(df, values_verify, columns_verify, nlanes, 'warmest')
