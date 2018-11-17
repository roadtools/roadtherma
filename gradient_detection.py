import networkx as nx
import numpy as np

TOLERANCE_RANGE_STEP = 0.5
tol_start, tol_end = (5, 15)
tolerances = np.arange(tol_start, tol_end, TOLERANCE_RANGE_STEP)

def calculate_tolerance_vs_percentage_high_gradient(df_temperature, nroad_pixels, offsets, tolerances):
    percentage_high_gradients = list()
    for tolerance in tolerances:
        high_gradients, _ = detect_high_gradient_pixels(df_temperature.values, offsets, tolerance)
        percentage_high_gradients.append((high_gradients.sum() / nroad_pixels) * 100)
    return percentage_high_gradients


def detect_high_gradient_pixels(temperatures, offsets, tolerance, diagonal_adjacency=True):
    """
    Return a boolean array the same size as `df_temperature` indexing all pixels
    having higher gradients than what is supplied in `config.gradient_tolerance`.
    The `offsets` list contains the road section identified for each transversal
    line (row) in the data.
    """
    gradient_map = np.zeros(temperatures.shape, dtype='bool')
    edges = []

    for idx in range(len(offsets) - 1):
        # Locate the temperature gradients in the driving direction
        longitudinal_edges = _detect_longitudinal_gradients(idx, offsets, temperatures, gradient_map, tolerance)
        edges.append(longitudinal_edges)

        # Locate the temperature gradients in the transversal direction
        tranversal_edges = _detect_transversal_gradients(idx, offsets, temperatures, gradient_map, tolerance)
        edges.append(tranversal_edges)

        if diagonal_adjacency:
            _detect_diagonal_gradients_right(idx, offsets, temperatures, gradient_map, tolerance)
            _detect_diagonal_gradients_left(idx, offsets, temperatures, gradient_map, tolerance)

    tranversal_edges = _detect_transversal_gradients(idx + 1, offsets, temperatures, gradient_map, tolerance)
    edges.append(tranversal_edges)

    gradient_graph = _create_gradient_graph(edges)
    clusters = list(_extract_clusters(gradient_graph))
    return gradient_map, clusters


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


def _detect_diagonal_gradients_right(idx, offsets, temperatures, gradient_map, tolerance):
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
    gradient_map[idx, start:end][indices] = 1
    gradient_map[idx + 1, next_start:next_end][indices] = 1

    edges = _calc_edges(idx, indices, idx + 1, indices + 1)
    return edges


def _detect_diagonal_gradients_left(idx, offsets, temperatures, gradient_map, tolerance):
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
    gradient_map[idx, start:end][indices] = 1
    gradient_map[idx + 1, next_start:next_end][indices] = 1

    edges = _calc_edges(idx, indices + 1, idx + 1, indices)
    return edges


def _detect_longitudinal_gradients(idx, offsets, temperatures, gradient_map, tolerance):
    start, end = offsets[idx]
    next_start, next_end = offsets[idx + 1]
    start = max(start, next_start)
    end = min(end, next_end)

    temperature_slice = temperatures[idx, start:end]
    temperature_slice_next = temperatures[idx + 1, start:end]

    (indices, ) = np.where(np.abs(temperature_slice - temperature_slice_next) > tolerance)
    indices += start
    gradient_map[idx, indices] = 1
    gradient_map[idx + 1, indices] = 1

    edges = _calc_edges(idx, indices, idx + 1, indices)
    return edges


def _detect_transversal_gradients(idx, offsets, temperatures, gradient_map, tolerance):
    start, end = offsets[idx]
    temperature_slice = temperatures[idx, start:end]
    (indices, ) = np.where(np.abs(np.diff(temperature_slice)) > tolerance)
    indices += start
    gradient_map[idx, indices] = 1
    gradient_map[idx, indices + 1] = 1
    edges = _calc_edges(idx, indices, idx, indices + 1)
    return edges
