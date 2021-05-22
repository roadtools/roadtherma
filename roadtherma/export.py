import datetime
import os

import numpy as np

from .utils import merge_temperature_data, split_temperature_data


def temperature_to_csv(file_path, temperatures, metadata, road_pixels):
    temperatures = temperatures.copy()
    filename = os.path.basename(file_path)
    temperatures.values[~road_pixels] = 'NaN'
    df = merge_temperature_data(metadata, temperatures)
    df.to_csv(f"gradient_{filename}.csv")


def detections_to_csv(file_path, detection_type, temperatures, metadata, road_pixels, detected_pixels):
    data_width = temperatures.values.shape[1]
    temperatures = temperatures.copy()
    filename = os.path.basename(file_path)
    temperatures.values[~road_pixels] = 0
    temperatures.values[road_pixels] = 1
    temperatures.values[detected_pixels & road_pixels] = 2
    temperatures[f'percentage_{detection_type}'] = 100 * (np.sum(detected_pixels & road_pixels, axis=1) / data_width)
    df = merge_temperature_data(metadata, temperatures)
    df.to_csv(f"area_{detection_type}_{filename}.csv")


def temperature_mean_to_csv(file_path, temperatures, road_pixels):
    temperatures = temperatures.copy()
    filename = os.path.basename(file_path)
    temperatures.values[~road_pixels] = 'NaN'
    temperatures['temperature_sum'] = np.nanmean(temperatures.values, axis=1)
    df = temperatures[['temperature_sum']]
    df.to_csv(f"distribution_{filename}.csv")


def clusters_to_csv(file_path, clusters):
    filename = os.path.basename(file_path)
    del clusters['coordinates']
    clusters.to_csv(f"gradient_cluster_stats_{filename}.csv")
