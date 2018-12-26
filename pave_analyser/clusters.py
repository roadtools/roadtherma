import numpy as np
import pandas as pd


def create_cluster_dataframe(data):
    clusters = data.clusters
    clusters_npixel = np.array([len(cluster) for cluster in clusters])
    clusters = [np.array(cluster) for cluster in clusters]
    df = pd.DataFrame.from_dict({
        'size_npixel':clusters_npixel,
        'coordinates':clusters
        })
    pixel_area = data.pixel_width * data.pixel_height
    df['size_m^2'] = df.size_npixel * pixel_area
    df['center_pixel'] = df['coordinates'].apply(_cluster_center)
    df['center_gps'] = df.apply(_center_gps, axis=1, args=(data,))
    df['center_chainage'] = df.apply(_center_chainage, axis=1, args=(data,))
    df['start_time'] = df.apply(_start_time, axis=1, args=(data.df.time,))
    df['end_time'] = df.apply(_end_time, axis=1, args=(data.df.time,))
    df['mean_temperature'] = df.apply(_mean_cluster_temperature, axis=1, args=(data.temperatures.values,))
    return df


def filter_clusters(df, sqm=None, npixels=None):
    if sqm is not None:
        df = df[df['size_m^2'] >= sqm]
    if npixels is not None:
        df = df[df['size_npixel'] >= npixels]
    return df.copy()


def _mean_cluster_temperature(cluster, temperature_map):
    temperatures = [
            temperature_map[row, col]
            for row, col in cluster.coordinates
            ]
    return np.mean(temperatures)


def _center_gps(row, data):
    idx, _ = row.center_pixel
    longitude = data.df.longitude[idx]
    latitude = data.df.latitude[idx]
    return (longitude, latitude)


def _center_chainage(row, data):
    idx, _ = row.center_pixel
    chainage = data.df.distance[idx]
    return chainage


def _cluster_center(coordinates):
    # This is calculated as a arithmetic mean of the coordinates which might be a bad metric
    x = int(round(coordinates[:, 0].mean()))
    y = int(round(coordinates[:, 1].mean()))
    return x,y


def _start_time(row, time):
    min_idx = min([idx for idx, _ in row.coordinates])
    return time[min_idx]


def _end_time(row, time):
    max_idx = max([idx for idx, _ in row.coordinates])
    return time[max_idx]
