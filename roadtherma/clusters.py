import numpy as np
import pandas as pd

from .utils import longitudinal_resolution

def create_cluster_dataframe(pixel_temperatures, clusters_raw, metadata, pixelwidth):
    clusters_npixel = np.array([len(cluster) for cluster in clusters_raw])
    clusters = [np.array(cluster) for cluster in clusters_raw]
    df = pd.DataFrame.from_dict({
        'size_npixel':clusters_npixel,
        'coordinates':clusters
        })
    pixel_area = pixelwidth * longitudinal_resolution(metadata.distance)
    df['size_m^2'] = df.size_npixel * pixel_area
    df['center_pixel'] = df['coordinates'].apply(_cluster_center)
    df['center_gps'] = df.apply(_center_gps, axis=1, args=(metadata,))
    df['center_chainage'] = df.apply(_center_chainage, axis=1, args=(metadata,))
    df['start_time'] = df.apply(_start_time, axis=1, args=(metadata.time,))
    df['end_time'] = df.apply(_end_time, axis=1, args=(metadata.time,))
    df['mean_temperature'] = df.apply(_mean_cluster_temperature, axis=1, args=(pixel_temperatures,))
    return df


def filter_clusters(clusters, gradient_pixels, sqm=None, npixels=None):
    df = clusters
    if sqm is not None:
        df = df[df['size_m^2'] >= sqm]
    if npixels is not None:
        df = df[df['size_npixel'] >= npixels]
    clusters = df.copy()

    # rebuild gradient_map based on the (possibly) reduced set of clusters
    gradient_pixels_new = np.zeros(gradient_pixels.shape, dtype='bool')
    clusters_raw_new = []
    for _idx, cluster in clusters.iterrows():
        coords = cluster.coordinates
        gradient_pixels_new[coords[:, 0], coords[:, 1]] = 1
        clusters_raw_new.append(coords.tolist())
    return gradient_pixels_new, clusters_raw_new


def _mean_cluster_temperature(cluster, temperature_map):
    temperatures = [
            temperature_map[row, col]
            for row, col in cluster.coordinates
            ]
    return np.mean(temperatures)


def _center_gps(row, metadata):
    idx, _ = row.center_pixel
    longitude = metadata.longitude.values[idx]
    latitude = metadata.latitude.values[idx]
    return (longitude, latitude)


def _center_chainage(row, metadata):
    idx, _ = row.center_pixel
    chainage = metadata.distance[idx]
    return chainage


def _cluster_center(coordinates):
    # This is calculated as a arithmetic mean of the coordinates which might be a bad metric
    x = int(round(coordinates[:, 0].mean()))
    y = int(round(coordinates[:, 1].mean()))
    return x,y


def _start_time(row, time):
    min_idx = min([idx for idx, _ in row.coordinates])
    return time.values[min_idx]


def _end_time(row, time):
    max_idx = max([idx for idx, _ in row.coordinates])
    return time.values[max_idx]
