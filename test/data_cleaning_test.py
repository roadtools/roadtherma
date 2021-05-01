import unittest
import numpy as np
import pandas as pd

from roadtherma.road_identification import clean_data, estimate_road_width, detect_paving_lanes
from roadtherma.data import create_road_pixels, create_trimming_result_pixels

class TestRoadWidthDetection(unittest.TestCase):
    threshold = 1.0
    adjust = 0
    tolerance = 0.1

    def test_roadwidth_detection(self):
        pixels = np.array([
            [0, 9, 9, 9, 9],
            [9, 9, 9, 9, 0]
            ])

        roadwidths = estimate_road_width(pixels, self.threshold, self.adjust, self.adjust)
        self.assertEqual(roadwidths, [(1, 5), (0, 4)])

        road_pixels = create_road_pixels(pixels, roadwidths)
        np.testing.assert_array_equal(
                road_pixels,
                np.array([
                    [0, 1, 1, 1, 1],
                    [1, 1, 1, 1, 0]
                    ])
                )


class TestLaneSelector(unittest.TestCase):
    threshold = 9.5

    def _execute_test(self, df, values_verify, columns_verify, nlanes, selector):
        lane_result = detect_paving_lanes(df, self.threshold)

        if nlanes == 1:
            assert lane_result['warmest'] == lane_result['coldest']
        else:
            assert lane_result['warmest'] != lane_result['coldest']

        lane_start, lane_end = lane_result[selector]
        df_after = df.iloc[:,lane_start:lane_end]
        self.assertEqual(
                columns_verify,
                list(df_after.columns)
                )
        np.testing.assert_array_equal(
                df_after.values,
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


class TestDataCleaningRegressionTest(unittest.TestCase):
    """
    This suite targets trim_temperature_data but is also looking at the result
    of create_trimming_result_pixels.
    """
    config = {
            'autotrim_temperature': 80,
            'autotrim_percentage': 25,
            'lane_threshold': 80,
            'lane_to_use': 'warmest',
            'roadwidth_threshold': 80,
            'roadwidth_adjust_left': 1,
            'roadwidth_adjust_right': 1,
            }

    def test_remove_first_and_last_rows(self):
        temperatures = pd.DataFrame(np.array([
            [ 0,  0, 99,  0,  0],
            [99, 99, 99, 99, 99],
            [ 0, 99,  0,  0,  0],
            ], dtype='float'), columns=['T1', 'T2', 'T3', 'T4', 'T5'])

        temperatures_trimmed, trim_result, lane_result, roadwidths = clean_data(
                temperatures, self.config)

        assert (0, 5, 1, 2) == trim_result
        assert [(1, 4)] == roadwidths

        pixel_category = create_trimming_result_pixels(
                temperatures.values, trim_result, lane_result['warmest'], roadwidths)

        np.testing.assert_array_equal(pixel_category, np.array([
            [ 0, 0, 0, 0, 0],
            [ 0, 1, 1, 1, 0],
            [ 0, 0, 0, 0, 0],
            ], dtype='int'))

    def test_remove_first_and_last_columns(self):
        temperatures = pd.DataFrame(np.array([
            [ 0, 99, 99, 99,  0],
            [99, 99, 99, 99, 99],
            [ 0, 99, 99, 99,  0],
            [ 0, 99, 99, 99,  0],
            ], dtype='float'), columns=['T1', 'T2', 'T3', 'T4', 'T5'])

        temperatures_trimmed, trim_result, lane_result, roadwidths = clean_data(
                temperatures, self.config)

        assert (1, 4, 0, 4) == trim_result
        assert [(1, 2), (1, 2), (1, 2), (1, 2)] == roadwidths

        pixel_category = create_trimming_result_pixels(
                temperatures.values, trim_result, lane_result['warmest'], roadwidths)

        np.testing.assert_array_equal(pixel_category, np.array([
            [ 0, 0, 1, 0, 0],
            [ 0, 0, 1, 0, 0],
            [ 0, 0, 1, 0, 0],
            [ 0, 0, 1, 0, 0],
            ], dtype='int'))
