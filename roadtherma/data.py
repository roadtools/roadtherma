import pickle

from . import readers
from .utils import split_temperature_data
from .road_identification import trim_temperature_data, estimate_road_length, detect_paving_lanes

_readers = {
        'voegele_example':readers._read_vogele_example,
        'voegele_M119':readers._read_vogele_M119,
        'voegele_M30':readers._read_vogele_M30,
        'voegele_taulov':readers._read_vogele_taulov,
        'TF_old':readers._read_TF_old,
        'TF_new':readers._read_TF_new,
        'moba':readers._read_moba,
        }


class PavementIRData:
    offsets = None
    road_pixels = None
    gradient_pixels = None

    def __init__(self, title, filepath, reader, transversal_resolution):
        self.title = title
        self.filepath = filepath
        self.reader = reader
        self.transversal_resolution = transversal_resolution
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
    def longitudinal_resolution(self):
        t = self.df.distance.diff().describe()
        longitudinal_resolution = t['50%']
        return longitudinal_resolution

    @property
    def mean_velocity(self):
        if 'velocity' in self.df.columns:
            return self.df.velocity.mean()
        return 'N/A'

    @property
    def nroad_pixels(self):
        if self.road_pixels is not None:
            return self.road_pixels.sum()
        return None
