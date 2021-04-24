import pickle
import numpy as np

from .readers import readers
from .utils import split_temperature_data


def load_data(filepath, reader):
    return readers[reader](filepath)


def create_trimming_result_pixels(pixels_raw, trim_result, lane_result, roadwidths):
    pixel_category = np.zeros(pixels_raw.shape, dtype='int')
    trim_col_start, trim_col_end, trim_row_start, trim_row_end = trim_result
    lane_start, lane_end = lane_result
    view = pixel_category[trim_row_start:trim_row_end, trim_col_start:trim_col_end][:, lane_start:lane_end]

    for longitudinal_idx, (road_start, road_end) in enumerate(roadwidths):
        view[longitudinal_idx, road_start - 1:road_end + 1] = 1
    return pixel_category


def create_road_pixels(pixels_trimmed, roadwidths):
    road_pixels = np.zeros(pixels_trimmed.shape, dtype='bool')
    for idx, (road_start, road_end) in enumerate(roadwidths):
        road_pixels[idx, road_start:road_end] = 1
    return road_pixels


def create_detect_result_pixels(pixels_trimmed, road_pixels, detection_pixels):
    pixel_category = np.zeros(pixels_trimmed.shape, dtype='int')
    pixel_category[~ road_pixels] = 1
    pixel_category[road_pixels] = 2
    pixel_category[detection_pixels] = 3
    return pixel_category
