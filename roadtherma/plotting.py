from functools import partial

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.ticker import FuncFormatter

from .utils import longitudinal_resolution
from .detections import detect_high_gradient_pixels


def categorical_heatmap(ax, pixels, distance, transversal_resolution, categories):
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


def aspect_ratio(data):
    nrows, ncols = data.temperatures.values.shape
    return (data.transversal_resolution * ncols) / (data.longitudinal_resolution * nrows)


def _set_meter_ticks_on_axes(ax, longitudinal_resolution, transversal_resolution, offset):
    format_x = partial(_distance_formatter, width=transversal_resolution)
    format_y = partial(_distance_formatter, width=longitudinal_resolution,
                       integer=True, offset=offset)
    formatter_x = FuncFormatter(format_x)
    formatter_y = FuncFormatter(format_y)
    ax.xaxis.set_major_formatter(formatter_x)
    ax.yaxis.set_major_formatter(formatter_y)
    ax.set_xlabel('Width [m]')


def temperature_heatmap(ax, pixels, distance, transversal_resolution):
    """Make a heatmap of the temperature columns in the dataframe."""
    _set_meter_ticks_on_axes(ax,
                             longitudinal_resolution(distance),
                             transversal_resolution,
                             distance.iloc[0]
                             )
    mat = ax.imshow(pixels, aspect="auto", cmap='RdYlGn_r')
    plt.colorbar(mat, ax=ax, label='Temperature [C]')


def create_map_of_analysis_results(data, method):
    map_ = data.temperatures.copy()
    map_.values[~ data.road_pixels] = 1
    map_.values[data.road_pixels] = 2

    if method == 'gradient':
        map_.values[data.gradient_pixels] = 3
    elif method == 'moving_average':
        map_.values[data.moving_average_pixels] = 3
    return map_.values


def plot_heatmaps(titles, metadata, transversal_resolution, pixel_temperatures, pixel_category, categories):
    fig_heatmaps, (ax1, ax2) = plt.subplots(ncols=2)
    fig_heatmaps.subplots_adjust(wspace=0.6)
    fig_heatmaps.suptitle(titles['main'])

    # Plot the raw data
    ax1.set_title(titles['temperature_title'])  # 'Raw data'
    temperature_heatmap(ax1, pixel_temperatures,
                        metadata.distance, transversal_resolution)
    ax1.set_ylabel('chainage [m]')

    # Plot that shows identified road and high gradient pixels
    ax2.set_title(titles['category_title'])  # 'Estimated high gradients'
    plt.figure(num=fig_heatmaps.number)
    # cat_array = create_map_of_analysis_results(data, method)
    # labels = ['Non-road', 'normal\nroad', 'high\ngradient\nroad']
    categorical_heatmap(ax2, pixel_category, metadata.distance,
                        transversal_resolution, categories)
    return fig_heatmaps


def _distance_formatter(x, _pos, width, offset=None, integer=False):
    if offset is None:
        offset = 0
    if integer:
        return '{}'.format(int(round(x*width + offset)))
    return '{:.1f}'.format(x*width + offset)


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


def _calculate_tolerance_vs_percentage_high_gradient(temperatures, roadwidths, road_pixels, tolerances):
    percentage_high_gradients = list()
    nroad_pixels = road_pixels.sum()
    for tolerance in tolerances:
        gradient_pixels, _ = detect_high_gradient_pixels(
            temperatures.values, roadwidths, tolerance)
        percentage_high_gradients.append(
            (gradient_pixels.sum() / nroad_pixels) * 100)
    return percentage_high_gradients


def save_figures(figures, n):
    for figure_name, figure in figures.items():
        plt.figure(num=figure.number)
        plt.savefig("{}{}.png".format(figure_name, n), dpi=500)
