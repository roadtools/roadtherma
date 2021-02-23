import numpy as np

from .utils import split_temperature_data, merge_temperature_data

def trim_temperature_data(data, threshold, autotrim_percentage):
    """
    Trim the temperature heatmap data by removing all outer rows and columns that only contains
    `autotrim_percentage` temperature values below `threshold`. The input `data` is modified inplace.

    :param data: Temperature data that should be trimmed.
    :type data: instance of :class:`.PavementIRData`
    :param float threshold: Temperature threshold used in data-trimming.
    :param float autotrim_percentage: Percentage of pixels in a row/column that is allowed to have temperatures
         above `threshold` and still be discarded.
    :return: Same as the `data` argument.
    """
    df = data.df.copy(deep=True)
    df_temperature, df_rest = split_temperature_data(df)
    df_temperature = _trim_temperature(df_temperature, threshold, autotrim_percentage)
    data.df = merge_temperature_data(df_temperature, df_rest)
    return data


def detect_paving_lanes(data, threshold, select='warmest'):
    """
    Detect lanes the one that is being actively paved during a two-lane paving operation where
    the lane that is not being paved during data acquisition has been recently paved and thus
    having a higher temperature compared to the surroundings.

    :param data: Temperature data where either one or two lanes should
        be detected.
    :type data: instance of :class:`.PavementIRData`
    :param float threshold: Temperature threshold used to detect both lanes
        from the surroundings. This means that the temperature should be sensitive
        enough to detect the coldest lane.
    :param str select: Lane selection method. `'warmest'` selects the lane with the
        highest mean temperature and `'lowest'` selects the lane lowest temperature.
    :return int: Number of lanes detected. This can be either 1 or 2.
    """
    df = data.df.copy(deep=True)
    df_temperature, df_rest = split_temperature_data(df)
    seperators = _calculate_lane_seperators(df_temperature.values, threshold)
    if seperators is None:
        return 1
    else:
        df_temperature = _select_lane(df_temperature, seperators, select)
        data.df = merge_temperature_data(df_temperature, df_rest)
        return 2


def estimate_road_length(data, threshold, adjust_left, adjust_right):
    """
    Estimate the road length of each transversal line (row) of the temperature
    heatmap data.

    :param data: Temperature data to perform road length detection on.
    :type data: instance of :class:`.PavementIRData`
    :param threshold: Threshold temperature used when classifying if pixels
        belongs to the road or not.
    :param int adjust_left: Additional number of pixels to cut off left edge after estimating road width.
    :param int adjust_right: Additional number of pixels to cut off right edge after estimating road width.

    :return: Same as the `data` argument.
    """
    pixels = data.temperatures.values
    offsets = []
    road_pixels = np.zeros(pixels.shape, dtype='bool')
    for idx in range(pixels.shape[0]):
        start = _estimate_road_edge_right(pixels[idx, :], threshold)
        end = _estimate_road_edge_left(pixels[idx, :], threshold)
        road_pixels[idx, start + adjust_left:end - adjust_right] = 1
        offsets.append((start + adjust_left, end - adjust_right))
    data.offsets = offsets
    data.road_pixels = road_pixels
    return data


def _calculate_lane_seperators(pixels, threshold):
    mean_temp = np.mean(pixels, axis=0)
    above_thresh = (mean_temp > threshold).astype('int')
    start = len(mean_temp) - len(np.trim_zeros(above_thresh, 'f'))
    end = - (len(mean_temp) - len(np.trim_zeros(above_thresh, 'b')))
    below_thresh = ~ above_thresh.astype('bool')
    if sum(below_thresh[start:end]) == 0:
        return None
    elif sum(below_thresh[start:end]) > 0:
        (midpoint, ) = np.where(mean_temp[start:end] == min(mean_temp[start:end]))
        midpoint = midpoint[0] + start
    return (start, midpoint, end)


def _select_lane(df_temperature, seperators, select):
    start, midpoint, end = seperators
    f_mean = df_temperature.iloc[:, start:midpoint].mean().mean()
    b_mean = df_temperature.iloc[:, midpoint + 1:end].mean().mean()
    columns = df_temperature.columns
    if f_mean > b_mean:
        warm_lane = columns[:midpoint + 1]
        cold_lane = columns[midpoint:] # We exclude the seperating column
    else:
        warm_lane = columns[midpoint:]
        cold_lane = columns[:midpoint + 1]

    if select == 'warmest':
        df_temperature = df_temperature[warm_lane]
    elif select == 'coldest':
        df_temperature = df_temperature[cold_lane]
    else:
        raise Exception('Unknown selection method "{}"'.format(select))
    return df_temperature


def _trim_temperature(df, autotrim_temperature, autotrim_percentage):
    df = _trim_temperature_columns(df, autotrim_temperature, autotrim_percentage)
    df = _trim_temperature_columns(df.T, autotrim_temperature, autotrim_percentage).T
    return df


def _trim_temperature_columns(df, threshold, autotrim_percentage):
    for column_name in df.columns:
        if not _trim(df, column_name, threshold, autotrim_percentage):
            break

    for column_name in reversed(df.columns):
        if not _trim(df, column_name, threshold, autotrim_percentage):
            break
    return df


def _trim(df, column, threshold_temp, autotrim_percentage):
    above_threshold = sum(df[column] > threshold_temp)
    above_threshold_pct = 100 * (above_threshold / len(df))
    if above_threshold_pct > autotrim_percentage:
        return False
    else:
        del df[column]
        return True


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
