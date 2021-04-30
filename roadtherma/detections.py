import networkx as nx
import numpy as np

from .utils import longitudinal_resolution

TOLERANCE_RANGE_STEP = 0.5
tol_start, tol_end = (5, 15)
tolerances = np.arange(tol_start, tol_end, TOLERANCE_RANGE_STEP)


def detect_temperatures_below_moving_average(temperatures, road_pixels, metadata, percentage=90, window_meters=100):
    """
    Returns a boolean array the same size as the temperature data, identifying all pixels
    with a temperature `percentage` lower than the moving average.
    The mo
    """
    temperature_pixels = temperatures.values
    window = round(window_meters / longitudinal_resolution(metadata.distance))
    moving_average = _calc_moving_average(temperatures, road_pixels, window)
    moving_average_values = np.tile(moving_average.values, (temperature_pixels.shape[1], 1)).T
    ratio = percentage / 100
    moving_average_pixels = temperature_pixels < (ratio * moving_average_values)
    moving_average_pixels[~ road_pixels] = 0  # non-road pixels are not part of the detection
    return moving_average_pixels


def _calc_moving_average(df, road_pixels, window):
    df = df.copy(deep=True)
    df.values[~ road_pixels] = 'NaN'
    min_periods = int(window * 0.80) # There must be 80% of <window> number of pixels
    df_avg = df.rolling(window, center=True, min_periods=min_periods).mean()
    moving_average = df_avg.mean(axis=1)
    return moving_average


def detect_high_gradient_pixels(temperature_pixels, roadwidths,  tolerance, diagonal_adjacency=True):
    """
    Return a boolean array the same size as `df_temperature` indexing all pixels
    having higher gradients than what is supplied in `config.gradient_tolerance`.
    The `roadwidths` list contains the road section identified for each transversal
    line (row) in the data.
    """
    gradient_pixels = np.zeros(temperature_pixels.shape, dtype='bool')
    edges = []

    for idx in range(len(roadwidths) - 1):
        # Locate the temperature gradients in the driving direction
        edges.append(_detect_longitudinal_gradients(idx, roadwidths, temperature_pixels, gradient_pixels, tolerance))

        # Locate the temperature gradients in the transversal direction
        edges.append(_detect_transversal_gradients(idx, roadwidths, temperature_pixels, gradient_pixels, tolerance))

        if diagonal_adjacency:
            edges.append(_detect_diagonal_gradients_right(idx, roadwidths, temperature_pixels, gradient_pixels, tolerance))
            edges.append(_detect_diagonal_gradients_left(idx, roadwidths, temperature_pixels, gradient_pixels, tolerance))

    edges.append(_detect_transversal_gradients(idx + 1, roadwidths, temperature_pixels, gradient_pixels, tolerance))

    gradient_graph = _create_gradient_graph(edges)
    clusters_raw = list(_extract_clusters(gradient_graph))
    return gradient_pixels, clusters_raw


def _iter_edges(edges):
    for edge_array in edges:
        for idx in range(edge_array.shape[1]):
            row1, col1, row2, col2 = edge_array[:, idx]
            yield (row1, col1), (row2, col2)


def _create_gradient_graph(edges):
    edge_iterator = _iter_edges(edges)
    gradient_graph = nx.from_edgelist(edge_iterator)
    return gradient_graph


def _extract_clusters(graph):
    for cluster in sorted(nx.connected_components(graph), key=len, reverse=True):
        cluster = [(int(row), int(col)) for row, col in cluster]
        yield cluster


def _calc_edges(rowidx1, colidx1, rowidx2, colidx2):
    edges = np.zeros((4, len(colidx2)))
    edges[0, :] = rowidx1
    edges[1, :] = colidx1
    edges[2, :] = rowidx2
    edges[3, :] = colidx2
    return edges


def _detect_diagonal_gradients_right(idx, offsets, temperatures, gradient_pixels, tolerance):
    start, end = offsets[idx]
    next_start, next_end = offsets[idx + 1]

    if  next_start < start:
        new_start = start
        new_next_start = start + 1
    elif start < next_start:
        new_start = next_start - 1
        new_next_start = next_start
    elif start == next_start:
        new_start = start
        new_next_start = next_start + 1

    if next_end < end:
        new_end = next_end - 1
        new_next_end = next_end
    elif end < next_end:
        new_end = end
        new_next_end = end + 1
    elif end == next_end:
        new_end = end - 1
        new_next_end = next_end

    next_start = new_next_start
    next_end = new_next_end
    start = new_start
    end = new_end


    temperature_slice = temperatures[idx, start:end]
    temperature_slice_next = temperatures[idx + 1, next_start:next_end]

    (indices, ) = np.where(np.abs(temperature_slice - temperature_slice_next) > tolerance)

    gradient_pixels[idx, start:end][indices] = 1
    gradient_pixels[idx + 1, next_start:next_end][indices] = 1

    edges = _calc_edges(idx, indices + start, idx + 1, indices + next_start)
    return edges


def _detect_diagonal_gradients_left(idx, offsets, temperatures, gradient_pixels, tolerance):
    start, end = offsets[idx]
    next_start, next_end = offsets[idx + 1]


    if  next_start < start:
        new_start = start
        new_next_start = start - 1
    elif start < next_start:
        new_start = next_start + 1
        new_next_start = next_start
    elif start == next_start:
        new_start = start + 1
        new_next_start = next_start


    if next_end < end:
        new_end = next_end + 1
        new_next_end = next_end
    elif end < next_end:
        new_end = end
        new_next_end = end - 1
    elif end == next_end:
        new_end = end
        new_next_end = end - 1

    next_start = new_next_start
    next_end = new_next_end
    start = new_start
    end = new_end


    temperature_slice = temperatures[idx, start:end]
    temperature_slice_next = temperatures[idx + 1, next_start:next_end]

    (indices, ) = np.where(np.abs(temperature_slice - temperature_slice_next) > tolerance)
    gradient_pixels[idx, start:end][indices] = 1
    gradient_pixels[idx + 1, next_start:next_end][indices] = 1

    edges = _calc_edges(idx, indices + start, idx + 1, indices + next_start)
    return edges


def _detect_longitudinal_gradients(idx, offsets, temperatures, gradient_pixels, tolerance):
    start, end = offsets[idx]
    next_start, next_end = offsets[idx + 1]
    start = max(start, next_start)
    end = min(end, next_end)

    temperature_slice = temperatures[idx, start:end]
    temperature_slice_next = temperatures[idx + 1, start:end]

    (indices, ) = np.where(np.abs(temperature_slice - temperature_slice_next) > tolerance)
    indices += start
    gradient_pixels[idx, indices] = 1
    gradient_pixels[idx + 1, indices] = 1

    edges = _calc_edges(idx, indices, idx + 1, indices)
    return edges


def _detect_transversal_gradients(idx, offsets, temperatures, gradient_pixels, tolerance):
    start, end = offsets[idx]
    temperature_slice = temperatures[idx, start:end]
    (indices, ) = np.where(np.abs(np.diff(temperature_slice)) > tolerance)
    indices += start
    gradient_pixels[idx, indices] = 1
    gradient_pixels[idx, indices + 1] = 1
    edges = _calc_edges(idx, indices, idx, indices + 1)
    return edges
