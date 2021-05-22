from functools import partial

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.ticker import FuncFormatter

from .data import create_detect_result_pixels
from .utils import longitudinal_resolution
from .detections import detect_high_gradient_pixels


def save_figures(figures, n):
    for figure_name, figure in figures.items():
        plt.figure(num=figure.number)
        plt.savefig("{}{}.png".format(figure_name, n), dpi=500)


def plot_detections(k, figures, **kwargs):
    config = kwargs['config']

    ## Perform and plot moving average detection results along with the trimmed temperature data
    if config['moving_average_enabled']:
        figures[f'moving_average{k}'] = _plot_moving_average_detection(**kwargs)

    ## Perform and plot gradient detection results along with the trimmed temperature data
    if config['gradient_enabled']:
        figures[f'gradient{k}'] = _plot_gradient_detection(**kwargs)


def plot_cleaning_results(config, metadata, temperatures, pixel_category):
    titles = {
        'main': config['title'] + " - data cleaning",
        'temperature_title': "raw temperature data",
        'category_title': "road detection results"
    }
    categories = ['non-road', 'road', 'roller']
    return _plot_heatmaps(
        titles,
        metadata,
        config['transversal_resolution'],
        temperatures.values,
        pixel_category,
        categories
    )


def plot_statistics(title, temperatures, roadwidths, road_pixels, tolerance):
    tol_start, tol_end, tol_step = tolerance
    tolerances = np.arange(tol_start, tol_end, tol_step)

    fig_stats, (ax1, ax2) = plt.subplots(ncols=2)
    fig_stats.suptitle(title)

    # Plot showing the percentage of road that is comprised of high gradient pixels for a given gradient tolerance
    high_gradients = _calculate_tolerance_vs_percentage_high_gradient(
        temperatures, roadwidths, road_pixels, tolerances)
    ax1.set_title('Percentage high gradient as a function of tolerance')
    ax1.set_xlabel('Threshold temperature difference [C]')
    ax1.set_ylabel('Percentage of road whith high gradient.')
    sns.lineplot(x=tolerances, y=high_gradients, ax=ax1)

    # Plot showing histogram of road temperature
    ax2.set_title('Road temperature distribution')
    ax2.set_xlabel('Temperature [C]')
    distplot_data = temperatures.values[road_pixels]
    sns.histplot(distplot_data, color="m", ax=ax2,
                 stat='density', discrete=True, kde=True)
    return fig_stats


def _plot_moving_average_detection(moving_average_pixels, config, temperatures_trimmed, road_pixels, metadata, **_kwargs):
    titles = {
        'main': config['title'] + " - moving average",
        'temperature_title': "Temperatures",
        'category_title': "Moving average detection results"
    }
    categories = ['non-road', 'road', 'detections']
    pixel_category = create_detect_result_pixels(
        temperatures_trimmed.values,
        road_pixels,
        moving_average_pixels
    )
    return _plot_heatmaps(
        titles,
        metadata,
        config['transversal_resolution'],
        temperatures_trimmed.values,
        pixel_category,
        categories
    )


def _plot_gradient_detection(gradient_pixels, config, temperatures_trimmed, metadata, road_pixels, **_kwargs):
    titles = {
        'main': config['title'] + " - gradient",
        'temperature_title': "Temperatures",
        'category_title': "gradient detection results"
    }
    categories = ['non-road', 'road', 'detections']
    pixel_category = create_detect_result_pixels(
        temperatures_trimmed.values,
        road_pixels,
        gradient_pixels
    )
    return _plot_heatmaps(
        titles,
        metadata,
        config['transversal_resolution'],
        temperatures_trimmed.values,
        pixel_category,
        categories
    )


def _calculate_tolerance_vs_percentage_high_gradient(temperatures, roadwidths, road_pixels, tolerances):
    percentage_high_gradients = list()
    nroad_pixels = road_pixels.sum()
    for tolerance in tolerances:
        gradient_pixels, _ = detect_high_gradient_pixels(
            temperatures.values, roadwidths, tolerance)
        percentage_high_gradients.append(
            (gradient_pixels.sum() / nroad_pixels) * 100)
    return percentage_high_gradients


def _plot_heatmaps(titles, metadata, transversal_resolution, pixel_temperatures, pixel_category, categories):
    fig_heatmaps, (ax1, ax2) = plt.subplots(ncols=2)
    fig_heatmaps.subplots_adjust(wspace=0.6)
    fig_heatmaps.suptitle(titles['main'])

    # Plot the raw data
    ax1.set_title(titles['temperature_title'])  # 'Raw data'
    _temperature_heatmap(ax1, pixel_temperatures,
                        metadata.distance, transversal_resolution)
    ax1.set_ylabel('chainage [m]')

    # Plot that shows identified road and high gradient pixels
    ax2.set_title(titles['category_title'])  # 'Estimated high gradients'
    plt.figure(num=fig_heatmaps.number)
    _categorical_heatmap(ax2, pixel_category, metadata.distance,
                        transversal_resolution, categories)
    return fig_heatmaps


def _categorical_heatmap(ax, pixels, distance, transversal_resolution, categories):
    _set_meter_ticks_on_axes(ax,
                             longitudinal_resolution(distance),
                             transversal_resolution,
                             distance.iloc[0]
                             )
    # set limits .5 outside true range
    colors = ["dimgray", "firebrick", "springgreen"]
    cmap = ListedColormap(colors[:len(categories)])
    mat = ax.imshow(pixels, aspect='auto', vmin=np.min(
        pixels) - .5, vmax=np.max(pixels) + .5, cmap=cmap)
    # tell the colorbar to tick at integers
    cbar = plt.colorbar(mat, ax=ax, ticks=np.arange(
        np.min(pixels), np.max(pixels) + 1))
    cbar.set_ticklabels(categories)


def _temperature_heatmap(ax, pixels, distance, transversal_resolution):
    """Make a heatmap of the temperature columns in the dataframe."""
    _set_meter_ticks_on_axes(ax,
                             longitudinal_resolution(distance),
                             transversal_resolution,
                             distance.iloc[0]
                             )
    mat = ax.imshow(pixels, aspect="auto", cmap='RdYlGn_r')
    plt.colorbar(mat, ax=ax, label='Temperature [C]')


def _set_meter_ticks_on_axes(ax, longitudinal_resolution, transversal_resolution, offset):
    format_x = partial(_distance_formatter, width=transversal_resolution)
    format_y = partial(_distance_formatter, width=longitudinal_resolution,
                       integer=True, offset=offset)
    formatter_x = FuncFormatter(format_x)
    formatter_y = FuncFormatter(format_y)
    ax.xaxis.set_major_formatter(formatter_x)
    ax.yaxis.set_major_formatter(formatter_y)
    ax.set_xlabel('Width [m]')


def _distance_formatter(x, _pos, width, offset=None, integer=False):
    if offset is None:
        offset = 0
    if integer:
        return '{}'.format(int(round(x*width + offset)))
    return '{:.1f}'.format(x*width + offset)
