import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import click

from .data import PavementIRData, cache_path, analyse_ir_data
from .utils import calculate_velocity, print_overall_stats
from .plotting import plot_statistics, plot_heatmaps, plot_heatmaps_section, save_figures
from .clusters import filter_clusters


matplotlib.rcParams.update({'font.size': 6})

@click.command()
@click.option('--plots/--no-plots', default=True, show_default=True,help='Whether or not to create plots.')
@click.option('--cache/--no-cache', default=True, show_default=True,help='Whether or not to use caching. If this is enabled and no caching files is found in "./.cache" the data will be processed from scratch and cached afterwards. The directory "./cache" must exist for caching to work.')
@click.option('--savefig/--no-savefig', default=False, show_default=True, help='Wheter or not to save the generated plots as png-files.')
@click.option('--trim_threshold', default=80.0, show_default=True, help='Temperature threshold for the data trimming step.')
@click.option('--percentage_above', default=0.2, show_default=True, help='Percentage of data that should be above trim_threshold in order for that outer longitudinal line to be removed.')
@click.option('--roadwidth_threshold', default=80.0, show_default=True, help='Temperature threshold for the road width estimation step.')
@click.option('--adjust_npixel', default=2, show_default=True, help='Additional number of pixels to cut off edges during road width estimation.')
@click.option('--gradient_tolerance', default=10.0, show_default=True, help='Tolerance on the temperature difference during temperature gradient detection.')
@click.option('--cluster_threshold_npixels', default=0, show_default=True, help='Minimum amount of pixels that should be in a cluster. Clusters below this value will be discarded.')
@click.option('--cluster_threshold_sqm', default=0.0, show_default=True, help='Minimum size of a cluster in square meters. Clusters below this value will be discarded.')
@click.option('--tolerance_range', nargs=3, default=(5, 20, 1), show_default=True, help='Range of tolerance values (e.g. "--tolerance_range <start> <end> <step size>") to use when plotting percentage of road that is comprised of high gradients vs gradient tolerance.')
def script(plots, cache, savefig, trim_threshold, percentage_above, roadwidth_threshold, adjust_npixel,
         gradient_tolerance, cluster_threshold_npixels, cluster_threshold_sqm, tolerance_range):
    """Command line tool for analysing Pavement IR data.
    It assumes that a file './data_files.py' (located where this script is executed)
    exists and contains a list of tuples named 'data_files' as follows:

        data_files = [\n
            (title, filepath, reader, width),\n
            ...,\n
            ]

    where 'title' is a string used as title in plots, 'filepath' contains the path
    of the particular data file, 'reader' is name of apropriate parser for that file
    and can have values "TF", "voegele_taulov", "voegele_example" or "voegele_M119".
    'width' is a float containing the resolution in meters of the pixels in the transversal
    direction (the horizontal resolution is derived from the chainage data).
    Options to configure the data processing is specified below.
    """
    tolerances = np.arange(*tolerance_range)

    namespace = {}
    exec(open('./data_files.py').read(), namespace)
    data_files = namespace['data_files']

    for n, (title, filepath, reader, pixel_width) in enumerate(data_files):
        print('Processing data file #{} - {}'.format(n, title))
        print('Path: {}'.format(filepath))
        cache_file_raw = cache_path(filepath, './.cache/{}_raw.pickle')
        cache_file = cache_path(filepath, './.cache/{}.pickle')
        if cache:
            data_raw = PavementIRData.from_cache(cache_file_raw)
            if data_raw is None:
                data_raw = PavementIRData(title, filepath, reader, pixel_width)
                data_raw.cache(cache_file_raw)
            data = PavementIRData.from_cache(cache_file)
            if data is None:
                data = analyse_ir_data(
                        data_raw, trim_threshold, percentage_above,
                        roadwidth_threshold, adjust_npixel, gradient_tolerance
                        )
                data.cache(cache_file)

        if not cache:
            data_raw = PavementIRData(title, filepath, reader, pixel_width)
            data = analyse_ir_data(
                    data_raw, trim_threshold, percentage_above,
                    roadwidth_threshold, adjust_npixel, gradient_tolerance
                    )

        calculate_velocity(data.df)
        filter_clusters(data, npixels=cluster_threshold_npixels, sqm=cluster_threshold_sqm)
        print_overall_stats(data)
        if plots:
            fig_stats = plot_statistics(title, data, tolerances)
            fig_heatmaps = plot_heatmaps(title, data, data_raw)
            fig_heatmaps_section = plot_heatmaps_section(title, data)
            figures = {
                    'fig_heatmaps':fig_heatmaps,
                    'fig_heatmaps_section':fig_heatmaps_section,
                    'fig_stats': fig_stats
                    }
            plt.show()
            if savefig:
                save_figures(figures, n)
    return data, data_raw


if __name__ == '__main__':
    script()
