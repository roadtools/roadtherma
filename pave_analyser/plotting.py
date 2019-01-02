from functools import partial

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

from .utils import calculate_tolerance_vs_percentage_high_gradient

def categorical_heatmap(ax, cat_array, data, labels, aspect='auto', cbar_kws=None):
    if cbar_kws is None:
        cbar_kws = dict()
    cmap = plt.get_cmap('magma', np.max(cat_array)-np.min(cat_array)+1)
    _set_meter_ticks_on_axes(ax, data)
    # set limits .5 outside true range
    mat = ax.imshow(cat_array, aspect=aspect, cmap=cmap,vmin = np.min(cat_array)-.5, vmax = np.max(cat_array)+.5)
    #tell the colorbar to tick at integers
    cbar = plt.colorbar(mat, ax=ax, ticks=np.arange(np.min(cat_array),np.max(cat_array) + 1), **cbar_kws)
    cbar.set_ticklabels(labels)


def aspect_ratio(data):
    nrows, ncols = data.temperatures.values.shape
    return (data.pixel_width * ncols) / (data.pixel_height * nrows)


def _set_meter_ticks_on_axes(ax, data):
    format_x = partial(_distance_formatter, width=data.pixel_width)
    format_y = partial(_distance_formatter, width=data.pixel_height,
            integer=True, offset=data.df.distance.iloc[0])
    formatter_x = FuncFormatter(format_x)
    formatter_y = FuncFormatter(format_y)
    ax.xaxis.set_major_formatter(formatter_x)
    ax.yaxis.set_major_formatter(formatter_y)
    ax.set_xlabel('Width [m]')


def temperature_heatmap(ax, data, aspect='auto', cmap='RdYlGn_r', cbar_kws=None, **kwargs):
    """Make a heatmap of the temperature columns in the dataframe."""
    if cbar_kws is None:
        cbar_kws = dict()
    _set_meter_ticks_on_axes(ax, data)
    mat = ax.imshow(data.temperatures.values, aspect=aspect, cmap=cmap)
    plt.colorbar(mat, ax=ax, **cbar_kws)


def create_map_of_analysis_results(data):
    map_ = data.temperatures.copy()
    map_.values[~ data.road_pixels] = 1
    map_.values[data.road_pixels] = 2
    map_.values[data.gradient_pixels] = 3
    return map_.values


def plot_heatmaps(title, data, data_raw):
    fig_heatmaps, (ax1, ax2, ax3) = plt.subplots(ncols=3)
    fig_heatmaps.subplots_adjust(wspace=0.6)
    #fig_heatmaps.tight_layout()
    fig_heatmaps.suptitle(title)

    ### Plot the raw data
    ax1.set_title('Raw data')
    temperature_heatmap(ax1, data_raw, cmap='RdYlGn_r', cbar_kws={'label':'Temperature [C]'})
    ax1.set_ylabel('chainage [m]')

    ### Plot trimmed data
    ax2.set_title('Trimmed data')
    temperature_heatmap(ax2, data, cmap='RdYlGn_r', cbar_kws={'label':'Temperature [C]'})

    ### Plot that shows identified road and high gradient pixels
    ax3.set_title('Estimated high gradients')
    plt.figure(num=fig_heatmaps.number)
    cat_array = create_map_of_analysis_results(data)
    labels = ['Non-road', 'normal\nroad', 'high\ngradient\nroad']
    categorical_heatmap(ax3, cat_array, data, labels)
    return fig_heatmaps


def _distance_formatter(x, pos, width, offset=None, integer=False):
    if offset is None:
        offset = 0
    if integer:
        return '{}'.format(int(round(x*width + offset)))
    else:
        return '{:.1f}'.format(x*width + offset)


def plot_heatmaps_section(title, data):
    data.resize(1000, 1100)
    aspect = aspect_ratio(data)
    fig_heatmaps, (ax1, ax2) = plt.subplots(ncols=2)
    fig_heatmaps.suptitle(title + ' (subsection)')

    ### Plot trimmed data
    ax1.set_title('Trimmed data')
    ax1.set_ylabel('Chainage [m]')
    temperature_heatmap(ax1, data, aspect=aspect, cbar_kws={'label':'Temperature [C]', 'shrink':0.7})

    ### Plot that shows identified road and high gradient pixels
    ax2.set_title('Estimated high gradients')
    cat_array = create_map_of_analysis_results(data)
    labels = ['Non-road', 'normal\nroad', 'high\ngradient\nroad']
    categorical_heatmap(ax2, cat_array, data, labels, aspect=aspect, cbar_kws={'shrink':0.7})
    return fig_heatmaps


def plot_single_cluster(data, cluster_no):
    fig_heatmaps, ax = plt.subplots(ncols=1)
    cat_array = data.temperatures.copy().values
    cat_array[~ data.road_pixels] = 1
    cat_array[data.road_pixels] = 2
    cat_array[data.gradient_pixels] = 3
    coords = data.clusters.coordinates.iloc[cluster_no]
    for row, col in coords:
        cat_array[row, col] = 4

    labels = ['Non-road', 'normal road', 'high gradient road', 'selected cluster']
    categorical_heatmap(ax, cat_array, data, labels, aspect='auto', cbar_kws=None)

def plot_statistics(title, data, tolerances):
    fig_stats, (ax1, ax2) = plt.subplots(ncols=2)
    fig_stats.suptitle(title)

    ### Plot showing the percentage of road that is comprised of high gradient pixels for a given gradient tolerance
    high_gradients = calculate_tolerance_vs_percentage_high_gradient(data, tolerances)
    ax1.set_title('Percentage high gradient as a function of tolerance')
    ax1.set_xlabel('Threshold temperature difference [C]')
    ax1.set_ylabel('Percentage of road whith high gradient.')
    sns.lineplot(x=tolerances, y=high_gradients, ax=ax1)

    ### Plot showing histogram of road temperature
    ax2.set_title('Road temperature distribution')
    ax2.set_xlabel('Temperature [C]')
    distplot_data = data.temperatures.values[data.road_pixels]
    sns.distplot(distplot_data, color="m", ax=ax2, norm_hist=False)
    return fig_stats


def save_figures(figures, n):
    for figure_name, figure in figures.items():
        plt.figure(num=figure.number)
        plt.savefig("{}{}.png".format(figure_name, n), dpi=500)
