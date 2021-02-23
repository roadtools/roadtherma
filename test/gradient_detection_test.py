import unittest

import numpy as np

from roadtherma.road_identification import estimate_road_length
from roadtherma.gradient_detection import _detect_longitudinal_gradients, _detect_transversal_gradients, detect_high_gradient_pixels

class DummyTemperatures:
    def __init__(self, array):
        self.values = array

class DummyData:
    pixel_width = 1
    pixel_height = 1

    def __init__(self, offsets, array):
        self.offsets = offsets
        self.temperatures = DummyTemperatures(array)

class TestGradientDetectionSimpleAdjacency(unittest.TestCase):
    tolerance = 9

    def test_longitudinal_detection(self):
        t = np.array([
                [10, 0, 0],
                [0, 10, 5]
                ])
        offsets = [[0, 3], [0, 3]]
        t_gradient = np.zeros(t.shape)
        _detect_longitudinal_gradients(0, offsets, t, t_gradient, self.tolerance)
        np.testing.assert_array_equal(
                t_gradient,
                np.array([
                    [1, 1, 0],
                    [1, 1, 0]
                    ])
                )

    def test_different_roadlengths(self):
        t = np.array([
                [10, 0, 0, 99],
                [0, 10, 5, 0 ],
                [99, 0, 0, 0 ],
                ])
        offsets = [[0, 3], [0, 4], [1, 4]]
        t_gradient = np.zeros(t.shape)
        _detect_longitudinal_gradients(0, offsets, t, t_gradient, self.tolerance)
        _detect_longitudinal_gradients(1, offsets, t, t_gradient, self.tolerance)
        np.testing.assert_array_equal(
                t_gradient,
                np.array([
                    [1, 1, 0, 0],
                    [1, 1, 0, 0],
                    [0, 1, 0, 0]
                    ])
                )


    def test_transversal_detection(self):
        t = np.array([
                [0, 10, 0],
                [0, 0, 10],
                [10, 0, 0],
                [0, 0, 0],
                ])
        offsets = [[0, 3], [0, 3], [0, 3], [0, 3]]
        t_gradient = np.zeros(t.shape)
        for n in range(t.shape[0]):
            _detect_transversal_gradients(n, offsets, t, t_gradient, self.tolerance)
        np.testing.assert_array_equal(
                t_gradient,
                np.array([
                    [1, 1, 1],
                    [0, 1, 1],
                    [1, 1, 0],
                    [0, 0, 0],
                    ])
                )


class TestGradientDetectionDiagonalAdjacency(unittest.TestCase):
    tolerance = 0.9
    offsets = [[0, 3], [0, 3], [0, 3]]

    def execute(self, array_input, array_verify, offsets=None):
        if offsets is None:
            offsets = self.offsets
        data = DummyData(offsets, array_input)
        detect_high_gradient_pixels(data, self.tolerance)
        np.testing.assert_array_equal(
                data.gradient_pixels,
                array_verify)


    def test_corner_right_diagonal_detection(self):
        array_input = np.array([
            [10, 0, 0],
            [0,  0, 0],
            [0,  0, 0]
            ])
        array_verify = np.array([
            [1, 1, 0],
            [1, 1, 0],
            [0, 0, 0]
            ])
        self.execute(array_input, array_verify)


    def test_corner_left_diagonal_detection(self):
        array_input = np.array([
            [0, 0,10],
            [0, 0, 0],
            [0, 0, 0]
            ])
        array_verify = np.array([
            [0, 1, 1],
            [0, 1, 1],
            [0, 0, 0]
            ])
        self.execute(array_input, array_verify)


    def test_middle_diagonal_detection(self):
        array_input = np.array([
            [0,  0, 0],
            [0, 10, 0],
            [0,  0, 0]
            ])
        array_verify = np.array([
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1]
            ])
        self.execute(array_input, array_verify)

    def test_uneven_roadwidth_large_width_differences(self):
        array_input = np.array([
            [0,  0, 0,  0, 0],
            [0, 10, 0, 10, 0],
            [0,  0, 0,  0, 0]
            ])
        offsets = [[0, 5], [0, 5], [0, 2]]
        array_verify = np.array([
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
            [1, 1, 0, 0, 0]
            ])
        self.execute(array_input, array_verify, offsets)


    def test_uneven_road_diagonal_case1(self):
        # left-diagonal, start_next < start
        array_input = np.array([
            [0, 2, 3],
            [4, 5, 6],
            ])
        offsets = [[1, 3], [0, 3]]
        array_verify = np.array([
            [0, 1, 1],
            [1, 1, 1],
            ])
        self.execute(array_input, array_verify, offsets)

    def test_uneven_road_diagonal_case2(self):
        # left-diagonal, start_next < start
        array_input = np.array([
            [1, 2, 3],
            [0, 5, 6],
            ])
        offsets = [[0, 3], [1, 3]]
        array_verify = np.array([
            [1, 1, 1],
            [0, 1, 1],
            ])
        self.execute(array_input, array_verify, offsets)

    def test_uneven_road_diagonal_case3(self):
        # left-diagonal, start_next == start
        array_input = np.array([
                [0, 2, 3],
                [0, 5, 6],
                ])
        offsets = [[1, 3], [1, 3]]
        array_verify = np.array([
            [0, 1, 1],
            [0, 1, 1],
            ])
        self.execute(array_input, array_verify, offsets)

    def test_uneven_road_diagonal_case4(self):
        # left-diagonal, end_next < end
        array_input = np.array([
            [1, 2, 3],
            [4, 5, 0],
            ])
        offsets = [[0, 3], [0, 2]]
        array_verify = np.array([
            [1, 1, 1],
            [1, 1, 0],
            ])
        self.execute(array_input, array_verify, offsets)

    def test_uneven_road_diagonal_case5(self):
        # left-diagonal, end < end_next
        array_input = np.array([
            [1, 2, 3],
            [4, 5, 0],
            ])
        offsets = [[0, 2], [0, 3]]
        array_verify = np.array([
            [1, 1, 0],
            [1, 1, 1],
            ])
        self.execute(array_input, array_verify, offsets)

    def test_uneven_road_diagonal_case6(self):
        # left-diagonal, end < end_next
        array_input = np.array([
            [1, 2, 0],
            [4, 5, 0],
            ])
        offsets = [[0, 2], [0, 2]]
        array_verify = np.array([
            [1, 1, 0],
            [1, 1, 0],
            ])
        self.execute(array_input, array_verify, offsets)


    def test_uneven_road_length_diagonal_corner1(self):
        array_input = np.array([
            [0,  0, 0],
            [0, 10, 0],
            [0,  0, 0]
            ])
        offsets = [[1, 3], [0, 3], [0, 2]]
        array_verify = np.array([
            [0, 1, 1],
            [1, 1, 1],
            [1, 1, 0]
            ])
        self.execute(array_input, array_verify, offsets)


    def test_uneven_road_length_diagonal_corner2(self):
        array_input = np.array([
            [0,  0, 0],
            [0, 10, 0],
            [0,  0, 0]
            ])
        offsets = [[0, 2], [0, 3], [1, 3]]
        array_verify = np.array([
            [1, 1, 0],
            [1, 1, 1],
            [0, 1, 1]
            ])
        self.execute(array_input, array_verify, offsets)


class TestClusterExtraction(unittest.TestCase):
    tolerance = 0.9
    offsets = [[0, 3], [0, 3], [0, 3]]

    def execute1(self, array_input, cluster_verify, cluster_verify_diag):
        data = DummyData(self.offsets, array_input)
        data = detect_high_gradient_pixels(data, self.tolerance, diagonal_adjacency=False)
        data_diag = DummyData(self.offsets, array_input)
        data_diag = detect_high_gradient_pixels(data_diag, self.tolerance, diagonal_adjacency=True)
        self.assertEqual(
                set(data.clusters_raw[0]),
                cluster_verify
                )
        self.assertEqual(
                set(data_diag.clusters_raw[0]),
                cluster_verify_diag
                )


    def test_clustering_edge(self):
        array_input = np.array([
            [0, 0, 0],
            [1, 0, 0],
            [0, 0, 0],
            ])
        cluster_verify = {(0, 0), (1, 0), (1,1), (2, 0)}
        cluster_verify_diag = {(0, 0), (0, 1), (1, 0), (1,1), (2, 0), (2, 1)}
        self.execute1(array_input, cluster_verify, cluster_verify_diag)


    def test_clustering_corner(self):
        array_input = np.array([
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 1],
            ])
        cluster_verify = {(1, 2), (2, 1), (2, 2)}
        cluster_verify_diag = {(1,1), (1, 2), (2, 1), (2, 2)}
        self.execute1(array_input, cluster_verify, cluster_verify_diag)


    def test_clustering_middle(self):
        array_input = np.array([
            [0, 0, 0],
            [0, 1, 0],
            [0, 0, 0],
            ])
        cluster_verify = {(0, 1), (1, 0), (1, 1), (1, 2), (2, 1)}
        cluster_verify_diag = {
                (0, 0), (0, 1), (0, 2),
                (1, 0), (1, 1), (1, 2),
                (2, 0), (2, 1), (2, 2)}
        self.execute1(array_input, cluster_verify, cluster_verify_diag)


    def test_clustering_two_clusters_no_diagonal(self):
        array_input = np.array([
            [2, 1, 0],
            [0, 0, 0],
            [0, 0, 1],
            ])
        data = DummyData(self.offsets, array_input)
        detect_high_gradient_pixels(data, self.tolerance, diagonal_adjacency=False)
        largest_cluster = set(data.clusters_raw[0])
        second_largest_cluster = set(data.clusters_raw[1])
        self.assertEqual(
                largest_cluster,
                {(0, 0), (0, 1), (0, 2), (1, 0), (1, 1)}
                )
        self.assertEqual(
                second_largest_cluster,
                {(1, 2), (2, 1), (2, 2)}
                )


    def test_clustering_two_clusters_with_diagonal(self):
        array_input = np.array([
            [2, 1, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 1],
            ])
        offsets = [[0, 3], [0, 3], [0, 3], [0, 3]]
        data = DummyData(offsets, array_input)
        detect_high_gradient_pixels(data, self.tolerance, diagonal_adjacency=True)
        largest_cluster = set(data.clusters_raw[0])
        second_largest_cluster = set(data.clusters_raw[1])
        self.assertEqual(
                largest_cluster,
                {(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)}
                )
        self.assertEqual(
                second_largest_cluster,
                {(2, 1), (2, 2), (3, 1), (3, 2)}
                )


    def test_large_cluster(self):
        threshold = 1.0
        adjust = 0
        tolerance = 0.1
        array_input = np.array([
            [0, 9, 8, 7, 9],
            [9, 8, 7, 8, 0]
            ])
        data = DummyData(None, array_input)
        estimate_road_length(data, threshold, adjust, adjust)
        detect_high_gradient_pixels(data, tolerance, diagonal_adjacency=True)
        self.assertEqual(data.offsets, [(1, 5), (0, 4)])
        np.testing.assert_array_equal(
                data.road_pixels,
                np.array([
                    [0, 1, 1, 1, 1],
                    [1, 1, 1, 1, 0]
                    ])
                )
        np.testing.assert_array_equal(
                data.gradient_pixels,
                np.array([
                    [0, 1, 1, 1, 1],
                    [1, 1, 1, 1, 0]
                    ])
                )
        self.assertEqual(set(data.clusters_raw[0]), {
            (0, 1), (0, 2), (0, 3), (0, 4),
            (1, 0), (1, 1), (1, 2), (1, 3)}
            )


if __name__ == '__main__':
    unittest.main()
