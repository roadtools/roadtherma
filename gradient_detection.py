import unittest
import numpy as np

from config import gradient_tolerance as tolerance


def detect_high_gradient_pixels(df_temperature, offsets):
    temperatures = df_temperature.values
    temperatures_gradient = np.zeros(temperatures.shape, dtype='bool')

    for idx in range(len(offsets) - 1):
        # Locate the temperature gradients in the driving direction
        _detect_longitudinal_gradients(idx, offsets, temperatures, temperatures_gradient, tolerance)

        # Locate the temperature gradients in the transversal direction
        _detect_transversal_gradients(idx, offsets, temperatures, temperatures_gradient, tolerance)
    return temperatures_gradient


def _detect_longitudinal_gradients(idx, offsets, temperatures, temperatures_gradient, tolerance):
    start, end = offsets[idx]
    next_start, next_end = offsets[idx + 1]
    start = max(start, next_start)
    end = min(end, next_end)

    temperature_slice = temperatures[idx, start:end]
    temperature_slice_next = temperatures[idx + 1, start:end]

    gradient_slice = temperatures_gradient[idx, start:end]
    gradient_slice_next = temperatures_gradient[idx + 1, start:end]

    high_gradient = np.abs(temperature_slice - temperature_slice_next) > tolerance

    gradient_slice[high_gradient] = 1
    gradient_slice_next[high_gradient] = 1


def _detect_transversal_gradients(idx, offsets, temperatures, temperatures_gradient, tolerance):
    start, end = offsets[idx]
    temperature_slice = temperatures[idx, start:end]
    gradient_slice = temperatures_gradient[idx, start:end]
    (indices, ) = np.where(np.abs(np.diff(temperature_slice)) > tolerance)
    gradient_slice[indices] = 1
    gradient_slice[indices + 1] = 1


class TestGradientDetection(unittest.TestCase):
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


if __name__ == '__main__':
    unittest.main()
