import numpy as np

from .utils import split_temperature_data, merge_temperature_data

def trim_temperature_data(data, threshold, percentage_above):
    """
    Trim the temperature heatmap dataframe such that all outer rows and columns that only contains
    `percentage_above` temperature values below `threshold` will be will be discarded.
    """
    df = data.df.copy(deep=True)
    df_temperature, df_rest = split_temperature_data(df)
    df_temperature = _trim_temperature(df_temperature, threshold, percentage_above)
    data.df = merge_temperature_data(df_temperature, df_rest)
    return data


def detect_paving_lanes(data, threshold, select='warmest'):
    """
    Detect lanes that is being actively paved during a two-lane paving operation where
    the lane not is not being paved during data acquisition has been recently paved, thus
    having a higher temperature compared to the surroundings.
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


def estimate_road_length(data, threshold, adjust_npixel):
    """
    Estimate the road length of each transversal section (row) of the road.
    Return a list of offsets for each row (transversal line).
    """
    pixels = data.temperatures.values
    offsets = []
    road_pixels = np.zeros(pixels.shape, dtype='bool')
    for idx in range(pixels.shape[0]):
        start = _estimate_road_edge_right(pixels[idx, :], threshold)
        end = _estimate_road_edge_left(pixels[idx, :], threshold)
        road_pixels[idx, start:end] = 1
        offsets.append((start + adjust_npixel, end - adjust_npixel))
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


def _trim_temperature(df, trim_threshold, percentage_above):
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
