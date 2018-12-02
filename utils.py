import pandas as pd
import re

from gradient_detection import detect_high_gradient_pixels

def split_temperature_data(df):
    """
    Splits a dataframe into two, one containing the temperature columns and one
    dataframe containing the rest. The rest could be chainage, gps-coordinates
    and so on.
    """
    temperature = temperature_columns(df)
    df_temperature = df[temperature]
    not_temperature = set(df.columns) - set(temperature)
    df_rest = df[list(not_temperature)]
    return df_temperature, df_rest


def merge_temperature_data(df_temp, df_rest):
    """
    Merge two dataframes containing temperature data and the rest, respectively, into a single dataframe.
    """
    return pd.merge(df_temp, df_rest, how='inner', copy=True, left_index=True, right_index=True)


def temperature_columns(df):
    """ Return a list containing the names of the temperature columns in the input dataframe."""
    temperature_columns = []
    for column in df.columns:
        if is_temperature.match(column) is not None:
            temperature_columns.append(column)
    return temperature_columns

is_temperature = re.compile('T\d+')


def calculate_velocity(df):
    dist_diff = df.distance.diff() # [meter]
    time_diff = df.time.diff() # [nanosecond]
    dist_diff[0] = dist_diff[1]
    time_diff[0] = time_diff[1]
    time_diff = time_diff.astype('int') / (1e9 * 60) # convert [nanosecond] -> [minute]
    df['velocity'] = dist_diff / time_diff # [meter / minute]


def calculate_tolerance_vs_percentage_high_gradient(data, tolerances):
    temperatures = data.temperatures.values
    percentage_high_gradients = list()
    for tolerance in tolerances:
        high_gradients, _ = detect_high_gradient_pixels(temperatures, data.offsets, tolerance)
        percentage_high_gradients.append((high_gradients.sum() / data.nroad_pixels) * 100)
    return percentage_high_gradients
