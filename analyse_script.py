import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from data import PavementIRData
from utils import calculate_velocity, temperature_columns
import config as cfg

def save_figures(figures):
    for figure_name, figure in figures.items():
        plt.figure(num=figure.number)
        plt.savefig("{}{}.png".format(figure_name, n), dpi=1200)#, dpi=800)


def categorical_heatmap(ax, data):
    cmap = plt.get_cmap('magma', np.max(data)-np.min(data)+1)
    # set limits .5 outside true range
    mat = ax.imshow(data, aspect='auto', cmap=cmap,vmin = np.min(data)-.5, vmax = np.max(data)+.5)
    #tell the colorbar to tick at integers
    cbar = plt.colorbar(mat, ticks=np.arange(np.min(data),np.max(data)+1))
    cbar.set_ticklabels(['lol1', 'lol2', 'lol3'])

def temperature_heatmap(df, **kwargs):
    """Make a heatmap of the temperature columns in the dataframe."""
    columns = temperature_columns(df)
    snsplot = sns.heatmap(df[columns], **kwargs)
    return snsplot

if __name__ == '__main__':
    for n, (title, filepath, reader) in enumerate(cfg.data_files):
        data = PavementIRData.from_cache(title, filepath, reader)
        #data = PavementIRData(title, filepath, reader)
        print('Processing data file #{} - {}'.format(n, title))
        if 'TF' not in title:
            # There is no timestamps in TF-data and thus no derivation of velocity
            calculate_velocity(data.df_raw)
            print('Mean paving velocity {:.1f} m/min'.format(data.df_raw.velocity.mean()))

        fig_heatmaps, (ax1, ax2, ax3) = plt.subplots(ncols=3)
        fig_heatmaps.suptitle(title)
        fig_stats, (ax4, ax5) = plt.subplots(ncols=2)
        fig_stats.suptitle(title)
        figures = {
                'fig_heatmaps':fig_heatmaps,
                'fig_stats': fig_stats
                }

        ### Plot the raw data
        ax1.set_title('Raw data')
        temperature_heatmap(data.df_temperature_raw, ax=ax1, cmap='RdYlGn_r', cbar_kws={'label':'Temperature [C]'})

        ### Plot trimmed data
        ax2.set_title('Trimmed data')
        temperature_heatmap(data.df_temperature, ax=ax2, cmap='RdYlGn_r', cbar_kws={'label':'Temperature [C]'})


        ### Plot showing the percentage of road that is comprised of high gradient pixels for a given gradient tolerance
        ax4.set_title('Percentage high gradient as a function of tolerance')
        sns.lineplot(x=cfg.tolerances, y=data.high_gradients, ax=ax4)

        ### Plot showing histogram of road temperature
        ax5.set_title('Road temperature distribution')
        distplot_data = data.df_temperature.values[~data.non_road_pixels]
        sns.distplot(distplot_data, color="m", ax=ax5, norm_hist=False)

        ### Plot that shows identified road and high gradient pixels
        ax3.set_title('Estimated high gradients')
        df_temperature = data.df_temperature.copy()
        df_temperature.values[data.non_road_pixels] = 1
        df_temperature.values[data.normal_road_pixels] = 2
        df_temperature.values[data.high_temperature_gradients] = 3
        plt.figure(num=fig_heatmaps.number)
        categorical_heatmap(ax3, df_temperature.values)

        plt.show()
        #save_figures(figures)
