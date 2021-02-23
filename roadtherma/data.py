import copy
import pickle
import pandas as pd

from .utils import split_temperature_data
from .road_identification import trim_temperature_data, estimate_road_length, detect_paving_lanes
from .gradient_detection import detect_high_gradient_pixels

def analyse_ir_data(
        data_raw, autotrim_temperature, autotrim_percentage, lane_threshold, roadwidth_threshold,
        roadwidth_adjust_left, roadwidth_adjust_right, gradient_tolerance, diagonal_adjacency=True):
    data = copy.deepcopy(data_raw)
    trim_temperature_data(data, autotrim_temperature, autotrim_percentage)
    detect_paving_lanes(data, lane_threshold, select='warmest')
    estimate_road_length(data, roadwidth_threshold, roadwidth_adjust_left, roadwidth_adjust_right)
    detect_high_gradient_pixels(data, gradient_tolerance, diagonal_adjacency)
    return data


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


def _read_vogele_M30(filename):
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
    _convert_vogele_timestamps(df, "%d/%m/%Y %H:%M:%S UTC + 02:00") # NOTE only difference between this and _read_vogele_taulov is the UTC-part here (ffs!)
    return df


_readers = {
        'TF':_read_TF,
        'voegele_example':_read_vogele_example,
        'voegele_M119':_read_vogele_M119,
        'voegele_M30':_read_vogele_M30,
        'voegele_taulov':_read_vogele_taulov
        }


def cache_path(filepath, template):
    *_, fname = filepath.split('/')
    return template.format(fname)


class PavementIRData:
    def __init__(self, title, filepath, reader, pixel_width):
        self.title = title
        self.filepath = filepath
        self.reader = reader
        self.pixel_width = pixel_width
        self.df = _readers[reader](filepath)

    @classmethod
    def from_file(cls, filepath):
        try:
            with open(filepath, 'rb') as f:
                return pickle.load(f)
        except FileNotFoundError:
            return None

    def to_file(self, filepath):
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)

    def resize(self, start, end):
        self.df = self.df[start:end]
        if self.offsets is not None:
            self.offsets = self.offsets[start:end]
        if self.road_pixels is not None:
            self.road_pixels = self.road_pixels[start:end]
        if self.gradient_pixels is not None:
            self.gradient_pixels = self.gradient_pixels[start:end]

    @property
    def temperatures(self):
        df_temperature, _ = split_temperature_data(self.df)
        return df_temperature

    @property
    def pixel_height(self):
        t = self.df.distance.diff().describe()
        pixel_height = t['50%']
        return pixel_height

    @property
    def mean_velocity(self):
        if 'velocity' in self.df.columns:
            return self.df.velocity.mean()
        else:
            return 'N/A'

    @property
    def nroad_pixels(self):
        if self.road_pixels is not None:
            return self.road_pixels.sum()
        return None
