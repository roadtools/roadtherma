import numpy as np

def trim_temperature(df, trim_threshold, percentage_above):
    """
    Trim the dataframe such that all outer rows and columns that only contains
    `percentage_above` temperature values below `threshold` will be will be discarded.
    """
    df = _trim_temperature_columns(df, trim_threshold, percentage_above)
    df = _trim_temperature_columns(df.T, trim_threshold, percentage_above).T
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


def estimate_road_length(pixels, threshold, adjust_npixel):
    """
    Estimate the road length of each transversal section (row) of the road.
    Return a list of offsets for each row (transversal line).
    """
    offsets = []
    road_pixels = np.zeros(pixels.shape, dtype='bool')
    for idx in range(pixels.shape[0]):
        start = _estimate_road_edge_right(pixels[idx, :], threshold)
        end = _estimate_road_edge_left(pixels[idx, :], threshold)
        road_pixels[idx, start:end] = 1
        offsets.append((start + adjust_npixel, end - adjust_npixel))
    return offsets, road_pixels


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
