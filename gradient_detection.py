import unittest
import numpy as np

TOLERANCE_RANGE_STEP = 0.5
tol_start, tol_end = (5, 15)
tolerances = np.arange(tol_start, tol_end, TOLERANCE_RANGE_STEP)

def calculate_tolerance_vs_percentage_high_gradient(df_temperature, nroad_pixels, offsets, tolerances):
    percentage_high_gradients = list()
    for tolerance in tolerances:
        high_gradients = detect_high_gradient_pixels(df_temperature, offsets, tolerance)
        percentage_high_gradients.append((high_gradients.sum() / nroad_pixels) * 100)
        #print('calculated percentage of high_gradient pixels (diff={}) of the road: {}'.format(tolerance, percentage_high_gradients[-1]))
    return percentage_high_gradients


def detect_high_gradient_pixels(df_temperature, offsets, tolerance):
    """
    Return a boolean array the same size as `df_temperature` indexing all pixels
    having higher gradients than what is supplied in `config.gradient_tolerance`.
    The `offsets` list contains the road section identified for each transversal
    line (row) in the data.
    """
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
