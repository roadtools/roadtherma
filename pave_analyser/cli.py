import numpy as np
import matplotlib.pyplot as plt
import click

from .data import PavementIRData, PavementIRDataRaw
from .utils import calculate_velocity
from .plotting import plot_statistics, plot_heatmaps, plot_heatmaps_section, save_figures

@click.command()
@click.option('--cache/--no-cache', default=True, show_default=True,help='Wheter or not to use caching. If this is enabled and no caching files is found in "./.cache" the data will be processed from scratch and cached afterwards. The directory "./cache" must exist for caching to work.')
@click.option('--savefig/--no-savefig', default=False, show_default=True, help='Wheter or not to save the generated plots as png-files.')
@click.option('--trim_threshold', default=80.0, show_default=True, help='Temperature threshold for the data trimming step.')
@click.option('--percentage_above', default=0.2, show_default=True, help='Percentage of data that should be above trim_threshold in order for that outer longitudinal line to be removed.')
@click.option('--roadwidth_threshold', default=80.0, show_default=True, help='Temperature threshold for the road width estimation step.')
@click.option('--adjust_npixel', default=2, show_default=True, help='Additional number of pixels to cut off edges during road width estimation.')
@click.option('--gradient_tolerance', default=10.0, show_default=True, help='Tolerance on the temperature difference during temperature gradient detection.')
@click.option('--tolerance_range', nargs=3, default=(5, 20, 1), show_default=True, help='Range of tolerance values (e.g. "--tolerance_range <start> <end> <step size>") to use when plotting percentage of road that is comprised of high gradients vs gradient tolerance.')
def script(cache, savefig, trim_threshold, percentage_above, roadwidth_threshold, adjust_npixel, gradient_tolerance, tolerance_range):
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
        if cache:
            data_raw = PavementIRDataRaw.from_cache(title, filepath)
            data = PavementIRData.from_cache(title, filepath)
            if data is None or data_raw is None:
                cache = False
        if not cache:
            data_raw = PavementIRDataRaw(title, filepath, reader, pixel_width)
            data = PavementIRData(data_raw, roadwidth_threshold, adjust_npixel, gradient_tolerance, trim_threshold, percentage_above)

        print('Processing data file #{} - {}'.format(n, title))
        if 'TF' not in title:
            # There is no timestamps in TF-data and thus no derivation of velocity
            calculate_velocity(data_raw.df)
            print('Mean paving velocity {:.1f} m/min'.format(data_raw.df.velocity.mean()))

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


if __name__ == '__main__':
    script()
