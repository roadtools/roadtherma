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


def clusters_to_csv(file_path, clusters):
    filename = file_path.split("/")[-1]
    del clusters['coordinates']
    clusters.to_csv(f"gradient_cluster_stats_{filename}.csv")
