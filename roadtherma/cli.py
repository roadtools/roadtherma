import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import click
import yaml

from .config import ConfigState
from .data import load_data, create_road_pixels, create_trimming_result_pixels, create_detect_result_pixels
from .utils import split_temperature_data
from .export import temperature_to_csv, detections_to_csv, temperature_mean_to_csv, clusters_to_csv
from .plotting import plot_statistics, plot_heatmaps, save_figures
from .clusters import create_cluster_dataframe
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
    figures = {}

    tol_start, tol_end, tol_step = config['tolerance']
    tolerances = np.arange(tol_start, tol_end, tol_step)

    print('Processing data file #{} - {}'.format(n, title))
    print('Path: {}'.format(file_path))
    df = load_data(file_path, config['reader'])
    temperatures, metadata = split_temperature_data(df)

    # Initial trimming of the dataset
    temperatures_trimmed, trim_result, lane_result, roadwidths = clean_data(temperatures, config)
    road_pixels = create_road_pixels(temperatures_trimmed.values, roadwidths)

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
    figures['fig_cleanup'] = plot_heatmaps(
        titles,
        metadata,
        config['transversal_resolution'],
        temperatures.values,
        pixel_category,
        categories
    )

    ## Perform and plot moving average detection results along with the trimmed temperature data
    if config['moving_average_enabled']:
        moving_average_pixels, figures['moving_average'] = moving_average_detection(
                config, temperatures_trimmed, road_pixels, metadata)

    ## Perform and plot gradient detection results along with the trimmed temperature data
    if config['gradient_enabled']:
        clusters_raw, figures['gradient'] = gradient_detection(
                config, temperatures_trimmed, metadata, road_pixels, roadwidths)

    # Plot statistics in relating to the gradient detection algorithm
    figures['stats'] = plot_statistics(
        title,
        temperatures_trimmed,
        roadwidths,
        road_pixels,
        tolerances
    )

    if config['write_csv']:
        temperature_to_csv(file_path, temperatures_trimmed, metadata, road_pixels)
        detections_to_csv(
                file_path, temperatures_trimmed, metadata, road_pixels, moving_average_pixels)
        temperature_mean_to_csv(file_path, temperatures_trimmed, road_pixels)
        if config['gradient_enabled']:
            clusters = create_cluster_dataframe(
                temperatures_trimmed.values,
                clusters_raw,
                metadata,
                config['transversal_resolution']
            )
            clusters_to_csv(file_path, clusters)

    if config['save_figures']:
        save_figures(figures, n)

    if config['show_plots']:
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


def moving_average_detection(config, temperatures_trimmed, road_pixels, metadata):
    moving_average_pixels = detect_temperature_difference(
        temperatures_trimmed,
        road_pixels,
        metadata,
        percentage=config['moving_average_percent'],
        window_meters=config['moving_average_window']
    )
    titles = {
        'main': config['title'],
        'temperature_title': "Result of moving average detection",
        'category_title': "Moving average detection results"
    }
    categories = ['non-road', 'road', 'detections']
    pixel_category = create_detect_result_pixels(
        temperatures_trimmed.values,
        road_pixels,
        moving_average_pixels
    )
    return moving_average_pixels, plot_heatmaps(
        titles,
        metadata,
        config['transversal_resolution'],
        temperatures_trimmed.values,
        pixel_category,
        categories
    )


def gradient_detection(config, temperatures_trimmed, metadata, road_pixels, roadwidths):
    gradient_pixels, clusters_raw = detect_high_gradient_pixels(
        temperatures_trimmed.values,
        roadwidths,
        config['gradient_tolerance'],
        diagonal_adjacency=True
    )
    titles = {
        'main': config['title'],
        'temperature_title': "result of gradient detection",
        'category_title': "gradient detection results"
    }
    categories = ['non-road', 'road', 'detections']
    pixel_category = create_detect_result_pixels(
        temperatures_trimmed.values,
        road_pixels,
        gradient_pixels
    )
    return clusters_raw, plot_heatmaps(
        titles,
        metadata,
        config['transversal_resolution'],
        temperatures_trimmed.values,
        pixel_category,
        categories
    )


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
