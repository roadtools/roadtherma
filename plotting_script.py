from functools import partial
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
import seaborn as sns

from data import PavementIRData, PavementIRDataRaw
from utils import calculate_velocity
import config as cfg

def save_figures(figures):
    for figure_name, figure in figures.items():
        plt.figure(num=figure.number)
        plt.savefig("{}{}.png".format(figure_name, n), dpi=1200)#, dpi=800)


def categorical_heatmap(ax, data, aspect='auto', cbar_kws=None):
    cat_array = create_map_of_analysis_results(data)

    if cbar_kws is None:
        cbar_kws = dict()
    cmap = plt.get_cmap('magma', np.max(cat_array)-np.min(cat_array)+1)
    set_meter_ticks_on_axes(ax, data)
    # set limits .5 outside true range
    mat = ax.imshow(cat_array, aspect=aspect, cmap=cmap,vmin = np.min(cat_array)-.5, vmax = np.max(cat_array)+.5)
    #tell the colorbar to tick at integers
    cbar = plt.colorbar(mat, ax=ax, ticks=np.arange(np.min(cat_array),np.max(cat_array) + 1), **cbar_kws)
    cbar.set_ticklabels(['Non-road', 'normal road', 'high gradient road'])


def aspect_ratio(data):
    nrows, ncols = data.temperatures.values.shape
    return (data.pixel_width * ncols) / (data.pixel_height * nrows)

def set_meter_ticks_on_axes(ax, data):
    format_x = partial(distance_formatter, width=data.pixel_width)
    format_y = partial(distance_formatter, width=data.pixel_height)
    formatter_x = FuncFormatter(format_x)
    formatter_y = FuncFormatter(format_y)
    ax.xaxis.set_major_formatter(formatter_x)
    ax.yaxis.set_major_formatter(formatter_y)
    ax.set_xlabel('width [m]')


def temperature_heatmap(ax, data, aspect='auto', cmap='RdYlGn_r', cbar_kws=None, **kwargs):
    """Make a heatmap of the temperature columns in the dataframe."""
    if cbar_kws is None:
        cbar_kws = dict()
    set_meter_ticks_on_axes(ax, data)
    mat = ax.imshow(data.temperatures.values, aspect=aspect, cmap=cmap)
    plt.colorbar(mat, ax=ax, **cbar_kws)

def create_map_of_analysis_results(data):
    df_temperature = data.temperatures.copy()
    df_temperature.values[data.non_road_pixels] = 1
    df_temperature.values[data.normal_road_pixels] = 2
    df_temperature.values[data.high_temperature_gradients] = 3
    return df_temperature.values

def plot_heatmaps(title, data, data_raw):
    fig_heatmaps, (ax1, ax2, ax3) = plt.subplots(ncols=3)
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
    categorical_heatmap(ax3, data)
    return fig_heatmaps

def distance_formatter(x, pos, width):
    return '{:.1f}'.format(x*width)


def plot_heatmaps_section(title, data):
    data.resize(1000, 1100)
    aspect = aspect_ratio(data)
    fig_heatmaps, (ax1, ax2) = plt.subplots(ncols=2)
    fig_heatmaps.suptitle(title + 'SECTION')

    ### Plot trimmed data
    ax1.set_title('Trimmed data')
    ax1.set_ylabel('chainage [m]')
    temperature_heatmap(ax1, data, aspect=aspect, cbar_kws={'label':'Temperature [C]', 'shrink':0.7})

    ### Plot that shows identified road and high gradient pixels
    ax2.set_title('Estimated high gradients')
    categorical_heatmap(ax2, data, aspect=aspect, cbar_kws={'shrink':0.7})
    return fig_heatmaps


def plot_statistics(title, data):
    fig_stats, (ax1, ax2) = plt.subplots(ncols=2)
    fig_stats.suptitle(title)

    ### Plot showing the percentage of road that is comprised of high gradient pixels for a given gradient tolerance
    ax1.set_title('Percentage high gradient as a function of tolerance')
    sns.lineplot(x=cfg.tolerances, y=data.high_gradients, ax=ax1)

    ### Plot showing histogram of road temperature
    ax2.set_title('Road temperature distribution')
    distplot_data = data.temperatures.values[~data.non_road_pixels]
    sns.distplot(distplot_data, color="m", ax=ax2, norm_hist=False)
    return fig_stats

if __name__ == '__main__':
    for n, (title, filepath, reader, pixel_width) in enumerate(cfg.data_files):
        data = PavementIRData.from_cache(title, filepath, reader, pixel_width)
        data_raw = PavementIRDataRaw.from_cache(title, filepath, reader, pixel_width)
        print('Processing data file #{} - {}'.format(n, title))
        if 'TF' not in title:
            # There is no timestamps in TF-data and thus no derivation of velocity
            calculate_velocity(data_raw.df)
            print('Mean paving velocity {:.1f} m/min'.format(data_raw.df.velocity.mean()))

        fig_stats = plot_statistics(title, data)
        fig_heatmaps = plot_heatmaps(title, data, data_raw)
        fig_heatmaps_section = plot_heatmaps_section(title, data)
        figures = {
                'fig_heatmaps':fig_heatmaps,
                'fig_heatmaps_section':fig_heatmaps_section,
                'fig_stats': fig_stats
                }
        plt.show()
        save_figures(figures)
