import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import click
import yaml

from .config import ConfigState
from .data import load_data, create_road_pixels, create_trimming_result_pixels
from .utils import split_temperature_data
from .export import temperature_to_csv, detections_to_csv, temperature_mean_to_csv, clusters_to_csv
from .plotting import plot_statistics, plot_detections, plot_cleaning_results, save_figures
from .clusters import create_cluster_dataframe
from .road_identification import clean_data, identify_roller_pixels, interpolate_roller_pixels
from .detections import detect_high_gradient_pixels, detect_temperatures_below_moving_average

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

    print('Processing data file #{} - {}'.format(n, title))
    print('Path: {}'.format(file_path))
    df = load_data(file_path, config['reader'])
    temperatures, metadata = split_temperature_data(df)

    # Initial trimming & cleaning of the dataset
    temperatures_trimmed, trim_result, lane_result, roadwidths = clean_data(temperatures, config)
    road_pixels = create_road_pixels(temperatures_trimmed.values, roadwidths)
    roller_pixels = identify_roller_pixels(
        temperatures_trimmed.values, road_pixels, config['roller_detect_temperature'])
    if config['roller_detect_interpolation']:
        interpolate_roller_pixels(temperatures_trimmed.values, roller_pixels, road_pixels)

    # Calculating detections
    moving_average_pixels = detect_temperatures_below_moving_average(
        temperatures_trimmed,
        road_pixels,
        metadata,
        percentage=config['moving_average_percent'],
        window_meters=config['moving_average_window']
    )
    gradient_pixels, clusters_raw = detect_high_gradient_pixels(
        temperatures_trimmed.values,
        roadwidths,
        config['gradient_tolerance'],
        diagonal_adjacency=True
    )

    # Plot trimming results
    pixel_category = create_trimming_result_pixels(
        temperatures.values, trim_result, lane_result[config['lane_to_use']],
        roadwidths, roller_pixels, config
    )
    for k, (start, end) in _iter_segments(temperatures, config['plotting_segments']):
        kwargs = {
            'config': config,
            'metadata': metadata.iloc[start:end, :],
            'temperatures': temperatures.iloc[start:end, :],
            'pixel_category': pixel_category[start:end, :],
            }
        figures[f'fig_cleanup{k}'] = plot_cleaning_results(**kwargs)

    # Plot detections results
    for k, (start, end) in _iter_segments(temperatures_trimmed, config['plotting_segments']):
        kwargs = {
            'config': config,
            'metadata': metadata.iloc[start:end, :],
            'temperatures_trimmed': temperatures_trimmed.iloc[start:end, :],
            'roadwidths': roadwidths[start:end],
            'road_pixels': road_pixels[start:end, :],
            'moving_average_pixels': moving_average_pixels[start:end, :],
            'gradient_pixels': gradient_pixels[start:end, :],
            }
        plot_detections(n, figures, **kwargs)

    # Plot statistics in relating to the gradient detection algorithm
    if config['gradient_statistics_enabled']:
        figures['stats'] = plot_statistics(
            title,
            temperatures_trimmed,
            roadwidths,
            road_pixels,
            config['tolerance']
        )


    # Export data through csv-files
    if config['write_csv']:
        temperature_to_csv(file_path, temperatures_trimmed, metadata, road_pixels)
        detections_to_csv(
                file_path, 'moving_avg', temperatures_trimmed, metadata, road_pixels, moving_average_pixels)
        detections_to_csv(
                file_path, 'gradient', temperatures_trimmed, metadata, road_pixels, gradient_pixels)
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


def _iter_segments(temperatures, number_of_segments):
    if int(number_of_segments) == 1:
        yield '', (0, len(temperatures))
        return

    segments = np.linspace(0, len(temperatures), number_of_segments + 1, dtype='int')
    yield from enumerate(zip(segments[:-1], segments[1:]))


if __name__ == '__main__':
    script()
