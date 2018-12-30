import datetime
import re
import pandas as pd

from .gradient_detection import detect_high_gradient_pixels

def print_overall_stats(data):
    report_template = """
========== REPORT ON THERMAL DATA ==========
Title of processed file: {title}
Location of processed file: {filepath}
Program finished running: {today}

---------- Overall Results ----------
Chainage start: {chainage_start} m
Chainage end: {chainage_end} m
Paving operation start: {pavetime_start}
Paving operation end: {pavetime_end}
Mean paving velocity: {mean_velocity:.1f}
Average paving temperature: {mean_paving_temp:.1f} C
Area of road with high gradient: {area_high_gradient:.1f} m²
Percentage of road with high gradient: {percentage_high_gradient:.1f}%
============================================
"""
    temperature_map = data.temperatures.values
    mean_paving_temp = temperature_map[data.road_pixels].mean()
    print(report_template.format(
        title=data.title,
        filepath=data.filepath,
        today=str(datetime.datetime.now()),
        chainage_start=data.df.distance.values[0],
        chainage_end=data.df.distance.values[-1],
        pavetime_start=data.df.time.min(),
        pavetime_end=data.df.time.max(),
        mean_velocity=data.mean_velocity,
        mean_paving_temp=mean_paving_temp,
        area_high_gradient=data.clusters['size_m^2'].sum(),
        percentage_high_gradient=(data.clusters.size_npixel.sum() / data.road_pixels.sum()) * 100
        ))

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
    if ('distance' in df.columns) and ('time' in df.columns):
        dist_diff = df.distance.diff().values # [meter]
        time_diff = df.time.diff().values # [nanosecond]
        dist_diff[0] = dist_diff[1]
        time_diff[0] = time_diff[1]
        time_diff = time_diff.astype('int') / (1e9 * 60) # convert [nanosecond] -> [minute]
        df['velocity'] = dist_diff / time_diff # [meter / minute]
        return True
    else:
        return False


def calculate_tolerance_vs_percentage_high_gradient(data, tolerances):
    percentage_high_gradients = list()
    for tolerance in tolerances:
        detect_high_gradient_pixels(data, tolerance)
        percentage_high_gradients.append((data.gradient_pixels.sum() / data.nroad_pixels) * 100)
    return percentage_high_gradients
