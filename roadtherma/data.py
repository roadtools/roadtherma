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


_readers = {
        'voegele_example':_read_vogele_example,
        'voegele_M119':_read_vogele_M119,
        'voegele_M30':_read_vogele_M30,
        'voegele_taulov':_read_vogele_taulov,
        'TF_old':_read_TF_old,
        'TF_new':_read_TF_new,
        'moba':_read_moba,
        }


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


def _read_TF_old(filename):
    temperatures = ['T{}'.format(n) for n in range(141)]
    columns = ['distance'] + temperatures + ['distance_again']
    df = pd.read_csv(filename, skiprows=7, delimiter=',', names=columns)
    del df['distance_again']
    del df['T140'] # This is the last column of the dataset (which is empty)
    return df


def _read_TF_new(filename):
    temperatures = ['T{}'.format(n) for n in range(281)]
    columns = ['distance']+['time']+['latitude']+ ['longitude']+ temperatures
    df = pd.read_csv(filename, skiprows=7, delimiter=',', names=columns)
    df['time']=[x[:-6] for x in df.time]
    df['time']=[x.replace(' ','') for x in df.time]
    df['time']=[x.replace('T',' ') for x in df.time]
    df['time'] = pd.to_datetime(df.time, format='%Y-%m-%d %H:%M:%S')
    del df['T280']
    return df

def _sensors_moba(filename):
    sensors = pd.read_csv(filename, sep = ';', skiprows=13, nrows=1)
    sensors.columns = ['name', 'number', 'none']
    n_sens = sensors.number + 1
    return n_sens

def _temperatures_moba(filename):
    sensors = pd.read_csv(filename, sep = ';', skiprows=13, nrows=1)
    sensors.columns = ['name', 'number', 'none']
    n_sens = sensors.number + 1
    temperatures_MOBA = ['T{}'.format(n) for n in range(int(n_sens))]
    return temperatures_MOBA

MOBA_BASE_COLUMNS = ['index','distance', 'speed','temporary_time', 'longitude', 'latitude']

def _rows_moba(filename):
    test = pd.read_csv(filename, sep = ';', skiprows=27)
    rows = 0
    for i in range(test.Index.size):
        if test.Index[i].isdigit() is True:
            rows += 1
        else:
            break
    return rows

def _read_moba(filename):
    with open(filename, newline='') as f:
        csv_reader = csv.reader(f)
        csv_headings = next(csv_reader)
        first_line = next(csv_reader)
    date=first_line[1].split(' ')
    date[0]=re.sub(r'\s+', '', first_line[2])
    date[1]=str(datetime.datetime.strptime(date[1], '%B').month)
    #date=date[::-1]
    date='-'.join(date)
    temperatures_MOBA= _temperatures_moba(filename)
    columns = MOBA_BASE_COLUMNS + ['signal_quality', 'satellites'] + temperatures_MOBA
    df = pd.read_csv(filename, skiprows=28,nrows=_rows_moba(filename), delimiter=';', names=columns, quoting=csv.QUOTE_NONE, quotechar='"', doublequote=True)
    df = df.drop(labels='index', axis=1)
    df['date']=date
    df['temporary_time']= df['date']+ [' '] + df['temporary_time']
    df = df.drop(labels='date', axis=1)
    df.insert(loc=2, column='time', value=[datetime.datetime.strptime(df['temporary_time'][i],'%Y-%m-%d %H:%M:%S')for i in range(df.shape[0])])
    df['time']=pd.to_datetime(df['time'])
    df = df.drop(labels='temporary_time', axis=1)
    df = df.drop(labels='speed', axis=1)
    df = df.drop(labels='satellites', axis=1)
    df = df[['time', 'distance','latitude', 'longitude', 'signal_quality']+ temperatures_MOBA]
    df[temperatures_MOBA]=df[temperatures_MOBA].apply(pd.to_numeric, errors='coerce', axis=1).fillna(0, downcast='infer')
    df = df.drop(labels=temperatures_MOBA[-1], axis=1)
    return df



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
