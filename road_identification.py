import numpy as np

import config as cfg

def trim_temperature(df):
    """
    Trim the dataframe such that all outer rows and columns that only contains
    `percentage_above` temperature values below `threshold` will be will be discarded.
    """
    df = _trim_temperature_columns(df, cfg.trim_threshold, cfg.percentage_above)
    df = _trim_temperature_columns(df.T, cfg.trim_threshold, cfg.percentage_above).T
    return df


def _trim_temperature_columns(df, threshold, percentage_above):
    for column_name in df.columns:
        if not _trim(df, column_name, threshold, percentage_above):
            break

    for column_name in reversed(df.columns):
        if not _trim(df, column_name, threshold, percentage_above):
            break
    return df


def _trim(df, column, threshold_temp, percentage_above):
    above_threshold = sum(df[column] > threshold_temp)
    above_threshold_pct = 100 * (above_threshold / len(df))
    if above_threshold_pct > percentage_above:
        return False
    else:
        del df[column]
        return True


def estimate_road_length(df, threshold):
    """
    Estimate the road length of each transversal section (row) of the road.
    Return a list of offsets for each row as well as a boolean array of indices
    indicating all non-road pixels.
    """
    values = df.values
    offsets = []
    non_road_pixels = np.ones(values.shape, dtype='bool')
    for distance_idx in range(values.shape[0]):
        offset_start = _estimate_road_edge_right(values[distance_idx, :], threshold)
        offset_end = _estimate_road_edge_left(values[distance_idx, :], threshold)
        non_road_pixels[distance_idx, offset_start:offset_end] = 0
        offsets.append((offset_start, offset_end))
    return offsets, non_road_pixels


def _estimate_road_edge_right(line, threshold):
    cond = line < threshold
    count = 0
    while True:
        if any(cond[count:count + 3]):
            count += 1
        else:
            break
    return count


def _estimate_road_edge_left(line, threshold):
    cond = line < threshold
    count = len(line)
    while True:
        if any(cond[count - 3:count]):
            count -= 1
        else:
            break
    return count
