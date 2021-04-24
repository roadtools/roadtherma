import numpy as np

from .utils import split_temperature_data, merge_temperature_data

def trim_temperature_data(pixels, threshold, autotrim_percentage):
    """
    Trim the temperature heatmap data by removing all outer rows and columns that only contains
    `autotrim_percentage` temperature values below `threshold`. The input `data` is modified inplace.

    :param pixels: Temperature heatmap :class:`numpy.ndarray` that should be trimmed.
    :type data: instance of :class:`.PavementIRData`
    :param float threshold: Temperature threshold used in data-trimming.
    :param float autotrim_percentage: Percentage of pixels in a row/column that is allowed to have temperatures
         above `threshold` and still be discarded.
    :return: Same as the `data` argument.
    """
    column_start, column_end = _trim_temperature_columns(pixels, threshold, autotrim_percentage)
    row_start, row_end = _trim_temperature_columns(pixels.T, threshold, autotrim_percentage)
    # df = merge_temperature_data(df_temperature.iloc[row_start:row_end, column_start:column_end], df_rest)
    return column_start, column_end, row_start, row_end


def _trim_temperature_columns(pixels, threshold, autotrim_percentage):
    pixel_start = 0
    pixel_end = -1

    for idx in range(pixels.shape[1]):
        if not _trim(pixels, idx, threshold, autotrim_percentage):
            break

        pixel_start = idx

    for idx in reversed(range(pixels.shape[1])):
        if not _trim(pixels, idx, threshold, autotrim_percentage):
            break

        pixel_end = idx

    return pixel_start, pixel_end


def _trim(pixels, column, threshold_temp, autotrim_percentage):
    above_threshold = sum(pixels[:, column] > threshold_temp)
    above_threshold_pct = 100 * (above_threshold / pixels.shape[0])
    if above_threshold_pct > autotrim_percentage:
        return False

    return True


def detect_paving_lanes(df, threshold):
    """
    Detect lanes the one that is being actively paved during a two-lane paving operation where
    the lane that is not being paved during data acquisition has been recently paved and thus
    having a higher temperature compared to the surroundings.

    :param df: Pavement data :class:`pd.DataFrame` with lanes to detect.
    :type data: instance of :class:`.PavementIRData`
    :param float threshold: Temperature threshold used to detect both lanes
        from the surroundings. This means that the temperature should be sensitive
        enough to detect the coldest lane.
    :return: "N"umber of lanes detected. This can be either 1 or 2.
    """
    df = df.copy(deep=True)
    df_temperature, _df_rest = split_temperature_data(df)
    pixels = df_temperature.values
    seperators = _calculate_lane_seperators(pixels, threshold)
    if seperators is None:
        lanes = {
                'warmest': (0, pixels.shape[1]),
                'coldest': (0, pixels.shape[1])
                }
    else:
        lanes = _classify_lanes(df_temperature.values, seperators)
    return lanes


def _calculate_lane_seperators(pixels, threshold):
    # mean for each longitudinal line:
    mean_temp = np.mean(pixels, axis=0)

    # Find the first longitudinal mean that is above threshold starting from each edge
    above_thresh = (mean_temp > threshold).astype('int')
    start = len(mean_temp) - len(np.trim_zeros(above_thresh, 'f'))
    end = - (len(mean_temp) - len(np.trim_zeros(above_thresh, 'b')))

    # If there are longitudinal means below temperature threshold in the middle
    # it is probably because there is a shift in lanes.
    below_thresh = ~ above_thresh.astype('bool')
    if sum(below_thresh[start:end]) == 0:
        return None

    if sum(below_thresh[start:end]) > 0:
        # Calculate splitting point between lanes
        (midpoint, ) = np.where(mean_temp[start:end] == min(mean_temp[start:end]))
        midpoint = midpoint[0] + start
        return (start, midpoint, end)
    return None


def _classify_lanes(pixels, seperators):
    start, midpoint, end = seperators
    f_mean = pixels[:, start:midpoint].mean()
    b_mean = pixels[:, midpoint + 1:end].mean()
    # columns = df_temperature.columns
    if f_mean > b_mean:
        warm_lane = (0, midpoint + 1)  # columns[:midpoint + 1]
        cold_lane = (midpoint, pixels.shape[1])  # columns[midpoint:] # We exclude the seperating column
    else:
        warm_lane = (midpoint, pixels.shape[1])
        cold_lane = (0, midpoint + 1)

    return {'warmest': warm_lane,
            'coldest': cold_lane}


def estimate_road_width(pixels, threshold, adjust_left, adjust_right):
    """
    Estimate the road length of each transversal line (row) of the temperature
    heatmap data.
    """
    road_widths = []
    for idx in range(pixels.shape[0]):
        start = _estimate_road_edge_right(pixels[idx, :], threshold)
        end = _estimate_road_edge_left(pixels[idx, :], threshold)
        road_widths.append((start + adjust_left, end - adjust_right))
    return road_widths


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
