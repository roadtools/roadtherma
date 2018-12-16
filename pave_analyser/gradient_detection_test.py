import unittest
import numpy as np

from gradient_detection import _detect_longitudinal_gradients, _detect_transversal_gradients, detect_high_gradient_pixels

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

    def test_corner_right_diagonal_detection(self):
        t = np.array([
                [10, 0, 0],
                [0,  0, 0],
                [0,  0, 0]
                ])
        gradient_map, _ = detect_high_gradient_pixels(t, self.offsets, self.tolerance)
        np.testing.assert_array_equal(
                gradient_map,
                np.array([
                    [1, 1, 0],
                    [1, 1, 0],
                    [0, 0, 0]
                    ])
                )


    def test_corner_left_diagonal_detection(self):
        t = np.array([
                [0, 0,10],
                [0, 0, 0],
                [0, 0, 0]
                ])
        gradient_map, _ = detect_high_gradient_pixels(t, self.offsets, self.tolerance)
        np.testing.assert_array_equal(
                gradient_map,
                np.array([
                    [0, 1, 1],
                    [0, 1, 1],
                    [0, 0, 0]
                    ])
                )


    def test_middle_diagonal_detection(self):
        t = np.array([
                [0,  0, 0],
                [0, 10, 0],
                [0,  0, 0]
                ])
        gradient_map, _ = detect_high_gradient_pixels(t, self.offsets, self.tolerance)
        np.testing.assert_array_equal(
                gradient_map,
                np.array([
                    [1, 1, 1],
                    [1, 1, 1],
                    [1, 1, 1]
                    ])
                )

    def test_uneven_roadwidth_large_width_differences(self):
        t = np.array([
                [0,  0, 0,  0, 0],
                [0, 10, 0, 10, 0],
                [0,  0, 0,  0, 0]
                ])

        offsets = [[0, 5], [0, 5], [0, 2]]
        gradient_map, _ = detect_high_gradient_pixels(t, offsets, self.tolerance)
        np.testing.assert_array_equal(
                gradient_map,
                np.array([
                    [1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1],
                    [1, 1, 0, 0, 0]
                    ])
                )


    def test_uneven_road_diagonal_case1(self):
        # left-diagonal, start_next < start
        t = np.array([
                [0, 2, 3],
                [4, 5, 6],
                ])
        offsets = [[1, 3], [0, 3]]
        gradient_map, _ = detect_high_gradient_pixels(t, offsets, self.tolerance)
        np.testing.assert_array_equal(
                gradient_map,
                np.array([
                    [0, 1, 1],
                    [1, 1, 1],
                    ])
                )

    def test_uneven_road_diagonal_case2(self):
        # left-diagonal, start_next < start
        t = np.array([
                [1, 2, 3],
                [0, 5, 6],
                ])
        offsets = [[0, 3], [1, 3]]
        gradient_map, _ = detect_high_gradient_pixels(t, offsets, self.tolerance)
        np.testing.assert_array_equal(
                gradient_map,
                np.array([
                    [1, 1, 1],
                    [0, 1, 1],
                    ])
                )

    def test_uneven_road_diagonal_case3(self):
        # left-diagonal, start_next == start
        t = np.array([
                [0, 2, 3],
                [0, 5, 6],
                ])
        offsets = [[1, 3], [1, 3]]
        gradient_map, _ = detect_high_gradient_pixels(t, offsets, self.tolerance)
        np.testing.assert_array_equal(
                gradient_map,
                np.array([
                    [0, 1, 1],
                    [0, 1, 1],
                    ])
                )

    def test_uneven_road_diagonal_case4(self):
        # left-diagonal, end_next < end
        t = np.array([
                [1, 2, 3],
                [4, 5, 0],
                ])
        offsets = [[0, 3], [0, 2]]
        gradient_map, _ = detect_high_gradient_pixels(t, offsets, self.tolerance)
        np.testing.assert_array_equal(
                gradient_map,
                np.array([
                    [1, 1, 1],
                    [1, 1, 0],
                    ])
                )

    def test_uneven_road_diagonal_case5(self):
        # left-diagonal, end < end_next
        t = np.array([
                [1, 2, 3],
                [4, 5, 0],
                ])
        offsets = [[0, 2], [0, 3]]
        gradient_map, _ = detect_high_gradient_pixels(t, offsets, self.tolerance)
        np.testing.assert_array_equal(
                gradient_map,
                np.array([
                    [1, 1, 0],
                    [1, 1, 1],
                    ])
                )

    def test_uneven_road_diagonal_case6(self):
        # left-diagonal, end < end_next
        t = np.array([
                [1, 2, 0],
                [4, 5, 0],
                ])
        offsets = [[0, 2], [0, 2]]
        gradient_map, _ = detect_high_gradient_pixels(t, offsets, self.tolerance)
        np.testing.assert_array_equal(
                gradient_map,
                np.array([
                    [1, 1, 0],
                    [1, 1, 0],
                    ])
                )


    def test_uneven_road_length_diagonal_corner1(self):
        t = np.array([
                [0,  0, 0],
                [0, 10, 0],
                [0,  0, 0]
                ])
        offsets = [[1, 3], [0, 3], [0, 2]]
        gradient_map, _ = detect_high_gradient_pixels(t, offsets, self.tolerance)
        np.testing.assert_array_equal(
                gradient_map,
                np.array([
                    [0, 1, 1],
                    [1, 1, 1],
                    [1, 1, 0]
                    ])
                )


    def test_uneven_road_length_diagonal_corner2(self):
        t = np.array([
                [0,  0, 0],
                [0, 10, 0],
                [0,  0, 0]
                ])
        offsets = [[0, 2], [0, 3], [1, 3]]
        gradient_map, _ = detect_high_gradient_pixels(t, offsets, self.tolerance)
        np.testing.assert_array_equal(
                gradient_map,
                np.array([
                    [1, 1, 0],
                    [1, 1, 1],
                    [0, 1, 1]
                    ])
                )


class TestClusterExtraction(unittest.TestCase):
    tolerance = 0.9
    offsets = [[0, 3], [0, 3], [0, 3]]

    def test_clustering_edge(self):
        t = np.array([
                [0, 0, 0],
                [1, 0, 0],
                [0, 0, 0],
                ])
        _, clusters = detect_high_gradient_pixels(t, self.offsets, self.tolerance, diagonal_adjacency=False)
        _, clusters_diag = detect_high_gradient_pixels(t, self.offsets, self.tolerance, diagonal_adjacency=True)
        self.assertEqual(
                set(clusters[0]),
                {(0, 0), (1, 0), (1,1), (2, 0)}
                )
        self.assertEqual(
                set(clusters_diag[0]),
                {(0, 0), (0, 1), (1, 0), (1,1), (2, 0), (2, 1)}
                )


    def test_clustering_corner(self):
        t = np.array([
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 1],
                ])
        _, clusters = detect_high_gradient_pixels(t, self.offsets, self.tolerance, diagonal_adjacency=False)
        _, clusters_diag = detect_high_gradient_pixels(t, self.offsets, self.tolerance, diagonal_adjacency=True)
        self.assertEqual(
                set(clusters[0]),
                {(1, 2), (2, 1), (2, 2)}
                )
        self.assertEqual(
                set(clusters_diag[0]),
                {(1,1), (1, 2), (2, 1), (2, 2)}
                )


    def test_clustering_middle(self):
        t = np.array([
                [0, 0, 0],
                [0, 1, 0],
                [0, 0, 0],
                ])
        _, clusters = detect_high_gradient_pixels(t, self.offsets, self.tolerance, diagonal_adjacency=False)
        _, clusters_diag = detect_high_gradient_pixels(t, self.offsets, self.tolerance, diagonal_adjacency=True)
        self.assertEqual(
                set(clusters[0]),
                {(0, 1), (1, 0), (1, 1), (1, 2), (2, 1)}
                )
        self.assertEqual(
                set(clusters_diag[0]),
                {(0, 0), (0, 1), (0, 2),
                 (1, 0), (1, 1), (1, 2),
                 (2, 0), (2, 1), (2, 2)}
                )


    def test_clustering_two_clusters_no_diagonal(self):
        t = np.array([
                [2, 1, 0],
                [0, 0, 0],
                [0, 0, 1],
                ])
        _, clusters = detect_high_gradient_pixels(t, self.offsets, self.tolerance, diagonal_adjacency=False)
        largest_cluster = set(clusters[0])
        second_largest_cluster = set(clusters[1])
        self.assertEqual(
                largest_cluster,
                {(0, 0), (0, 1), (0, 2), (1, 0), (1, 1)}
                )
        self.assertEqual(
                second_largest_cluster,
                {(1, 2), (2, 1), (2, 2)}
                )


    def test_clustering_two_clusters_with_diagonal(self):
        t = np.array([
                [2, 1, 0],
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 1],
                ])
        offsets = [[0, 3], [0, 3], [0, 3], [0, 3]]
        _, clusters = detect_high_gradient_pixels(t, offsets, self.tolerance, diagonal_adjacency=True)
        largest_cluster = set(clusters[0])
        second_largest_cluster = set(clusters[1])
        self.assertEqual(
                largest_cluster,
                {(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)}
                )
        self.assertEqual(
                second_largest_cluster,
                {(2, 1), (2, 2), (3, 1), (3, 2)}
                )


    def test_large_cluster(self):
        from road_identification import estimate_road_length
        threshold = 1.0
        adjust_npixel = 0
        tolerance = 0.1
        pixels = np.array([
            [0, 9, 8, 7, 9],
            [9, 8, 7, 8, 0]
            ])
        offsets, road_pixels = estimate_road_length(pixels, threshold, adjust_npixel)
        gradient_map, clusters = detect_high_gradient_pixels(pixels, offsets, tolerance, diagonal_adjacency=True)
        self.assertEqual(offsets, [(1, 5), (0, 4)])
        np.testing.assert_array_equal(
                road_pixels,
                np.array([
                    [0, 1, 1, 1, 1],
                    [1, 1, 1, 1, 0]
                    ])
                )
        np.testing.assert_array_equal(
                gradient_map,
                np.array([
                    [0, 1, 1, 1, 1],
                    [1, 1, 1, 1, 0]
                    ])
                )
        self.assertEqual(set(clusters[0]), {
            (0, 1), (0, 2), (0, 3), (0, 4),
            (1, 0), (1, 1), (1, 2), (1, 3)}
            )


if __name__ == '__main__':
    unittest.main()
