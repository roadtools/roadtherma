import matplotlib.pyplot as plt
import numpy as np

from data import import_TF, import_vogele_taulov, import_vogele_M119
from utils import estimate_road_length, trim_temperature, plot_data, split_temperature_data, merge_temperature_data, calculate_velocity
from gradient_detection import detect_high_gradient_pixels
import config as cfg

if __name__ == '__main__':
    data = [import_vogele_taulov(), import_vogele_M119()] + list(import_TF())
    for n, (title, df) in enumerate(data):
        print('Processing data file #{} - {}'.format(n, title))
        if 'TF' not in title:
            # There is no timestamps in TF-data and thus no derivation of velocity
            calculate_velocity(df)
            print('Mean paving velocity {:.1f} m/min'.format(df.velocity.mean()))

        fig, (ax1, ax2, ax3) = plt.subplots(ncols=3)
        fig.suptitle(title)
        df_temperature, df_rest = split_temperature_data(df)

        ### Plot the raw data
        ax1.set_title('Raw data')
        plot_data(df_temperature, ax=ax1, cmap='RdYlGn_r', cbar_kws={'label':'Temperature [C]'})

        ### Plot trimmed data
        ax2.set_title('Trimmed data')
        df_temperature = trim_temperature(df_temperature)
        plot_data(df_temperature, ax=ax2, cmap='RdYlGn_r', cbar_kws={'label':'Temperature [C]'})

        ### Estimate what is actual road and which pixels have are part of high temperature gradients
        offsets, non_road_pixels = estimate_road_length(df_temperature, cfg.roadlength_threshold)
        high_temperature_gradients = detect_high_gradient_pixels(df_temperature, offsets)
        normal_road_pixels = ~ (high_temperature_gradients | non_road_pixels) # Pixels identified as road without high temperature gradients
        df_temperature.values[high_temperature_gradients] = np.max(df_temperature.values[normal_road_pixels]) + 20
        df_temperature.values[non_road_pixels] = np.min(df_temperature.values[normal_road_pixels]) - 20
        ax3.set_title('Estimated high gradients')
        plot_data(df_temperature, ax=ax3, cmap='magma', cbar_kws={'label':'Black: Not road, Bright Yellow: High gradients detected'})

        ### Merge the processed temperature data with the rest of the dataset
        df = merge_temperature_data(df_temperature, df_rest)
        #plt.savefig("trimmed_data{}.png".format(n), dpi=800)
        plt.show()
