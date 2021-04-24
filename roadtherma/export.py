import datetime

import numpy as np

from .utils import merge_temperature_data, split_temperature_data


def temperature_to_csv(file_path, temperatures, metadata, road_pixels):
    temperatures = temperatures.copy()
    filename = file_path.split("/")[-1]
    temperatures.values[~road_pixels] = 'NaN'
    df = merge_temperature_data(metadata, temperatures)
    df.to_csv(f"gradient_{filename}.csv")


def detections_to_csv(file_path, temperatures, metadata, road_pixels, detected_pixels):
    temperatures = temperatures.copy()
    filename = file_path.split("/")[-1]
    temperatures.values[~road_pixels] = 0
    temperatures.values[road_pixels] = 1
    temperatures.values[detected_pixels & road_pixels] = 2
    temperatures['percentage_gradient'] = 100 * (np.sum(detected_pixels & road_pixels, axis=1) / np.sum(road_pixels, axis=1))
    df = merge_temperature_data(metadata, temperatures)
    df.to_csv(f"area_{filename}.csv")


def temperature_mean_to_csv(file_path, temperatures, road_pixels):
    temperatures = temperatures.copy()
    filename = file_path.split("/")[-1]
    temperatures.values[~road_pixels] = 'NaN'
    temperatures['temperature_sum'] = np.nanmean(temperatures.values, axis=1)
    df = temperatures[['temperature_sum']]
    df.to_csv(f"distribution_{filename}.csv")


def print_overall_stats(data):
    report_template = """
============= REPORT ON THERMAL DATA =============
Title:{title}
Location of file: {filepath}
Program finished running: {today}

---------------- Overall Results -----------------
Chainage start: {chainage_start} m
Chainage end:   {chainage_end} m
Paving operation start: {pavetime_start}
Paving operation end:   {pavetime_end}
Mean paving velocity:   {mean_velocity:.1f} m/min
Average paving temperature:         {mean_paving_temp:.1f} C
Area of road with high gradient:    {area_high_gradient:.1f} m²
Percentage road with high gradient: {percentage_high_gradient:.1f}%"""
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


def print_cluster_stats(data):
    header = "----------------- Cluster Stats -----------------"
    footer = "=================================================="
    template = \
"""Number of pixels: {npixels}
Total area:       {area:.1f} m²
Mean temperature: {temperature:.1f} C
Mean Chainage:    {chainage:.1f} m
Time start:       {time_start}
Time end:         {time_end}
Mean GPS: {gps}
-------------------------------------------------"""
    print(header)
    for _idx, cluster in data.clusters.iterrows():
        print(template.format(
            npixels=cluster.size_npixel,
            area=cluster['size_m^2'],
            temperature=cluster.mean_temperature,
            chainage=cluster.center_chainage,
            gps=cluster.center_gps,
            time_start=cluster.start_time,
            time_end=cluster.end_time
            ))
    print(footer)
