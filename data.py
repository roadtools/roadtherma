import pickle
import pandas as pd

from utils import split_temperature_data, merge_temperature_data
from road_identification import trim_temperature, estimate_road_length
from gradient_detection import detect_high_gradient_pixels, calculate_tolerance_vs_percentage_high_gradient
import config as cfg

def _read_TF(filename):
    temperatures = ['T{}'.format(n) for n in range(141)]
    columns = ['distance'] + temperatures + ['distance_again']
    df = pd.read_csv(filename, skiprows=7, delimiter=',', names=columns)
    del df['distance_again']
    del df['T140'] # This is the last column of the dataset which is empty
    return df


temperatures_voegele = ['T{}'.format(n) for n in range(52)]
VOEGELE_BASE_COLUMNS = ['time', 'distance', 'latitude', 'longitude']


def _convert_vogele_timestamps(df, formatting):
    df['time'] = pd.to_datetime(df.time, format=formatting)


def _read_vogele_example(filename):
    """ Old example code. This is probably not going to be used. """
    columns = VOEGELE_BASE_COLUMNS + temperatures_voegele
    df = pd.read_csv(filename, skiprows=2, delimiter=';', names=columns, decimal=',')
    _convert_vogele_timestamps(df, "%d.%m.%Y %H:%M:%S")
    return df


def _read_vogele_M119(filename):
    """
    Data similar to the example file.
    NOTE removed last line in the file as it only contained 'Keine Daten vorhanden'.
    """
    columns = VOEGELE_BASE_COLUMNS + ['signal_quality'] + temperatures_voegele
    df = pd.read_csv(filename, skiprows=2, delimiter=';', names=columns, decimal=',')
    _convert_vogele_timestamps(df, "%d.%m.%Y %H:%M:%S")
    return df


def _read_vogele_taulov(filename):
    """
    NOTE removed last line in the file as it only contained 'No data to display'.
    """
    import csv
    columns = VOEGELE_BASE_COLUMNS + ['signal_quality'] + temperatures_voegele
    df = pd.read_csv(filename, skiprows=3, delimiter=',', names=columns, quoting=csv.QUOTE_NONE, quotechar='"', doublequote=True)
    for col in df.columns:
        if col == 'time':
            df[col] = df[col].apply(lambda x:x.strip('"'))
        if col in set(temperatures_voegele) | {'distance', 'latitude', 'longitude'}:
            df[col] = df[col].astype('str').apply(lambda x:x.strip('"')).astype('float')
    _convert_vogele_timestamps(df, "%d/%m/%Y %H:%M:%S")
    return df


_readers = {
        'TF':_read_TF,
        'voegele_example':_read_vogele_example,
        'voegele_M119':_read_vogele_M119,
        'voegele_taulov':_read_vogele_taulov
        }


def _cache_path(filepath):
    cache_path = './.cache/{}.pickle'
    *_, fname = filepath.split('/')
    return cache_path.format(fname)


def _trim_data(df):
    df = df.copy(deep=True)
    df_temperature, df_rest = split_temperature_data(df)
    df_temperature = trim_temperature(df_temperature)
    return merge_temperature_data(df_temperature, df_rest)


def _identify_road(df):
    df_temperature, df_rest = split_temperature_data(df)
    offsets, non_road_pixels = estimate_road_length(df_temperature, cfg.roadlength_threshold)
    return offsets, non_road_pixels


class PavementIRData:
    def __init__(self, title, filepath, reader, pixel_width, cache=True):
        self.pixel_width = pixel_width
        self.title = title
        self.filepath = filepath

        ### Load the data and perform initial trimming
        self.df_raw = _readers[reader](filepath)
        self.df = _trim_data(self.df_raw)
        self.offsets, self.non_road_pixels = _identify_road(self.df)

        ### Perform gradient detection
        temperatures = self.temperature_data()
        self.high_temperature_gradients = detect_high_gradient_pixels(temperatures, self.offsets, cfg.gradient_tolerance)
        self.high_gradients = calculate_tolerance_vs_percentage_high_gradient(temperatures, self.nroad_pixels, self.offsets, cfg.tolerances)
        if cache:
            self.cache()

    @classmethod
    def from_cache(cls, title, filepath, reader, pixel_width):
        try:
            with open(_cache_path(filepath), 'rb') as f:
                return pickle.load(f)
        except FileNotFoundError:
            return cls(title, filepath, reader, pixel_width)

    def cache(self):
        with open(_cache_path(self.filepath), 'wb') as f:
            pickle.dump(self, f)

    def temperature_data(self, raw=False):
        if raw:
            df_temperature, _ = split_temperature_data(self.df_raw)
        else:
            df_temperature, _ = split_temperature_data(self.df)
        return df_temperature

    @property
    def nroad_pixels(self):
        return self.road_pixels.sum()

    @property
    def road_pixels(self):
        return ~self.non_road_pixels

    @property
    def normal_road_pixels(self):
        return (~ self.high_temperature_gradients) & self.road_pixels # Pixels identified as road without high temperature gradients


if __name__ == '__main__':
    from config import data_files
    for data in data_files:
        ir_data = PavementIRData(*data)
