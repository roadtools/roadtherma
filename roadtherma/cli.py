import copy

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import click
import yaml

from .config import ConfigState
from .data import load_data, create_road_pixels, create_trimming_result_pixels, create_detect_result_pixels
from .utils import calculate_velocity, split_temperature_data
from .export import temperature_to_csv, detections_to_csv, temperature_mean_to_csv
from .plotting import plot_statistics, plot_heatmaps, save_figures #, plot_heatmaps_sections
from .clusters import filter_clusters, create_cluster_dataframe
from .road_identification import trim_temperature_data, estimate_road_width, detect_paving_lanes
from .detections import detect_high_gradient_pixels, detect_temperature_difference

matplotlib.rcParams.update({'font.size': 6})


@click.command()
@click.option('--jobs_file', default="./jobs.yaml", show_default=True,
              help='path of the job specification file (in YAML format).')
def script(jobs_file):
    """Command line tool for analysing Pavement IR data.
    See https://github.com/roadtools/roadtherma for documentation on how to use
    this command line tool.
    """
    with open(jobs_file) as f:
        jobs_yaml = f.read()
    jobs = yaml.load(jobs_yaml, Loader=yaml.FullLoader)
    config_state = ConfigState()
    for n, item in enumerate(jobs):
        config = config_state.update(item['job'])
        process_job(n, config)


def process_job(n, config):
    title = config['title']
    file_path = config['file_path']

    tol_start, tol_end, tol_step = config['tolerance']
    tolerances = np.arange(tol_start, tol_end, tol_step)

    print('Processing data file #{} - {}'.format(n, title))
    print('Path: {}'.format(file_path))
    df = load_data(file_path, config['reader'])
    temperatures, metadata = split_temperature_data(df)
    temperatures_trimmed, trim_result, lane_result, roadwidths = clean_data(temperatures, config)

    # Initial trimming of the dataset
    ## Plot the data trimming and road identifcation results
    titles = {
        'main': title,
        'temperature_title': "raw temperature data",
        'category_title': "road detection results"
    }
    categories = ['non-road', 'road'] #, 'roller']
    pixel_category = create_trimming_result_pixels(
        temperatures.values,
        trim_result,
        lane_result[config['lane_to_use']],
        roadwidths
    )
    fig_cleanup = plot_heatmaps(
        titles,
        metadata,
        config['transversal_resolution'],
        temperatures.values,
        pixel_category,
        categories
    )
    # plt.show()


    ## Perform weakness detection using the two methods
    gradient_pixels, clusters_raw = detect_high_gradient_pixels(
        temperatures_trimmed.values,
        roadwidths,
        config['gradient_tolerance'],
        diagonal_adjacency=True
    )
    road_pixels = create_road_pixels(temperatures_trimmed.values, roadwidths)
    moving_average_pixels = detect_temperature_difference(
        temperatures_trimmed,
        road_pixels,
        metadata,
        percentage=config['moving_average_percent'],
        window_meters=config['moving_average_window']
    )


    ## Plot the detection results along with the trimmed temperature data
    # FIXME, using moving_average now. Should be configurable
    titles = {
        'main': title,
        'temperature_title': "result of moving average detection",
        'category_title': "moving average detection results"
    }
    categories = ['non-road', 'road', 'detections']
    pixel_temperatures = temperatures_trimmed.values
    pixel_category = create_detect_result_pixels(
        pixel_temperatures,
        road_pixels,
        moving_average_pixels
    )
    fig_detections = plot_heatmaps(
        titles,
        metadata,
        config['transversal_resolution'],
        pixel_temperatures,
        pixel_category,
        categories
    )

    # Plot statistics in relating to the gradient detection algorithm
    fig_stats = plot_statistics(
        title,
        temperatures_trimmed,
        roadwidths,
        road_pixels,
        tolerances
    )
    plt.show()

    if config['write_csv']:
        temperature_to_csv(file_path, temperatures_trimmed, metadata, road_pixels)
        detections_to_csv(file_path, temperatures_trimmed, metadata, road_pixels, moving_average_pixels)
        temperature_mean_to_csv(file_path, temperatures_trimmed, road_pixels)

    if config['print_stats']:
        clusters = create_cluster_dataframe(
            pixel_temperatures,
            clusters_raw,
            metadata,
            config['transversal_resolution']
        )
        filter_clusters(
            clusters,
            gradient_pixels,
            npixels=config['cluster_npixels'],
            sqm=config['cluster_sqm']
        )
        calculate_velocity(metadata)
        #print_overall_stats(data)
        #print_cluster_stats(data)


    # Save plots
    figures = {
        'fig_cleanup': fig_cleanup,
        'fig_detections': fig_detections,
        'fig_stats': fig_stats
    }

    if config['save_figures']:
        save_figures(figures, n)
    else:
        plt.show()

    for fig in figures.values():
        plt.close(fig)


def clean_data(temperatures, config):
    trim_result = trim_temperature_data(
        temperatures.values,
        config['autotrim_temperature'],
        config['autotrim_percentage']
    )
    column_start, column_end, row_start, row_end = trim_result
    temperatures_trimmed = temperatures.iloc[row_start:row_end, column_start:column_end]

    lane_result = detect_paving_lanes(
        temperatures_trimmed,
        config['lane_threshold']
    )
    lane_start, lane_end = lane_result[config['lane_to_use']]


    temperatures_trimmed = temperatures_trimmed.iloc[:, lane_start:lane_end]

    roadwidths = estimate_road_width(
        temperatures_trimmed.values,
        config['roadwidth_threshold'],
        config['roadwidth_adjust_left'],
        config['roadwidth_adjust_right']
    )
    return temperatures_trimmed, trim_result, lane_result, roadwidths


def _iter_segments(df, df_raw, segment_width):
    start = df_raw.distance.min()
    distance_max = df_raw.distance.max()
    while True:
        end = start + segment_width
        df_section = df[start <= df.distance & df.distance < end].copy()
        df_raw_section = df_raw[start <= df_raw.distance & df_raw.distance < end].copy()
        yield df_raw_section, df_section
        if end >= distance_max:
            break

        start = end


if __name__ == '__main__':
    script()
