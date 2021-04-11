import copy

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import click
import yaml

from .data import PavementIRData
from .utils import calculate_velocity, print_overall_stats, print_cluster_stats
from .plotting import plot_statistics, plot_heatmaps, save_figures #, plot_heatmaps_sections
from .clusters import filter_clusters, create_cluster_dataframe
from .road_identification import trim_temperature_data, estimate_road_length, detect_paving_lanes
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
    for n, item in enumerate(jobs):
        process_job(n, item['job'])


def process_job(n, job):
    title = job['title']
    file_path = job['file_path']
    reader = job['reader']
    create_plots = job.setdefault('create_plots', True)
    save_figures_ = job.setdefault('save_figures', True)
    print_stats = job.setdefault('print_stats', True)
    transversal_resolution = job.setdefault('transversal_resolution', 0.25)
    autotrim_temperature = job.setdefault('autotrim_temperature', 80.0)
    autotrim_percentage = job.setdefault('autotrim_percentage', 0.2)
    lane_threshold = job.setdefault('lane_threshold', 110.0)
    roadwidth_threshold = job.setdefault('roadwidth_threshold', 80.0)
    roadwidth_adjust_left = job.setdefault('roadwidth_adjust_left', 2)
    roadwidth_adjust_right = job.setdefault('roadwidth_adjust_right', 2)
    gradient_tolerance = job.setdefault('gradient_tolerance', 10.0)
    moving_average_window = job.setdefault('moving_average_window', 100.0)
    moving_average_percent = job.setdefault('moving_average_percent', 90.0)
    cluster_npixels = job.setdefault('cluster_npixels', 0)
    cluster_sqm = job.setdefault('cluster_sqm', 0.0)
    tolerance = job.setdefault('tolerance', [5, 20, 1])

    print('Processing data file #{} - {}'.format(n, title))
    print('Path: {}'.format(file_path))
    data_raw = PavementIRData(title, file_path, reader, transversal_resolution)

    data = copy.deepcopy(data_raw)
    trim_temperature_data(data, autotrim_temperature, autotrim_percentage)
    detect_paving_lanes(data, lane_threshold, select='warmest')
    estimate_road_length(data, roadwidth_threshold,
                         roadwidth_adjust_left, roadwidth_adjust_right)
    detect_high_gradient_pixels(data, gradient_tolerance, True)
    detect_temperature_difference(data, percentage=moving_average_percent, window_meters=moving_average_window)

    if print_stats:
        create_cluster_dataframe(data)
        filter_clusters(data, npixels=cluster_npixels, sqm=cluster_sqm)
        calculate_velocity(data.df)
        print_overall_stats(data)
        print_cluster_stats(data)

    if create_plots:
        tolerances = np.arange(*tolerance)
        fig_stats = plot_statistics(title, data, tolerances)
        fig_heatmaps = plot_heatmaps(title, data, data_raw, method='moving_average')
        # This requires manual setting of index parameters.
        # fig_heatmaps_section = plot_heatmaps_section(title, data)
        figures = {
            'fig_heatmaps': fig_heatmaps,
            # 'fig_heatmaps_section':fig_heatmaps_section,
            'fig_stats': fig_stats
        }

        if save_figures_:
            save_figures(figures, n)
        else:
            plt.show()
        for fig in figures.values():
            plt.close(fig)
    return data, data_raw


if __name__ == '__main__':
    script()
