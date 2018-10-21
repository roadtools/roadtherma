import pandas as pd

import config as cfg

def import_TF():
    for title, file_ in cfg.TF_M14_files:
        temperatures = ['T{}'.format(n) for n in range(141)]
        columns = ['distance'] + temperatures + ['distance_again']
        df = pd.read_csv(file_, skiprows=7, delimiter=',', names=columns)
        del df['distance_again']
        del df['T140'] # This is the last column of the dataset which is empty
        yield title, df


temperatures_voegele = ['T{}'.format(n) for n in range(52)]
VOEGELE_BASE_COLUMNS = ['time', 'distance', 'latitude', 'longitude']

def _convert_vogele_timestamps(df, formatting):
    df['time'] = pd.to_datetime(df.time, format=formatting)


def import_vogele_example():
    """ Old example code. This is probably not going to be used. """
    columns = VOEGELE_BASE_COLUMNS + temperatures_voegele
    title, filename = cfg.voegele_example
    df = pd.read_csv(filename, skiprows=2, delimiter=';', names=columns, decimal=',')
    _convert_vogele_timestamps(df, "%d.%m.%Y %H:%M:%S")
    return title, df


def import_vogele_M119():
    """
    Data similar to the example file.
    NOTE removed last line in the file as it only contained 'Keine Daten vorhanden'.
    """
    columns = VOEGELE_BASE_COLUMNS + ['signal_quality'] + temperatures_voegele
    title, filename = cfg.voegele_M119
    df = pd.read_csv(filename, skiprows=2, delimiter=';', names=columns, decimal=',')
    _convert_vogele_timestamps(df, "%d.%m.%Y %H:%M:%S")
    return title, df


def import_vogele_taulov():
    """
    NOTE removed last line in the file as it only contained 'No data to display'.
    """
    import csv
    columns = VOEGELE_BASE_COLUMNS + ['signal_quality'] + temperatures_voegele
    title, filename = cfg.voegele_taulov
    df = pd.read_csv(filename, skiprows=3, delimiter=',', names=columns, quoting=csv.QUOTE_NONE, quotechar='"', doublequote=True)
    for col in df.columns:
        if col == 'time':
            df[col] = df[col].apply(lambda x:x.strip('"'))
        if col in set(temperatures_voegele) | {'distance', 'latitude', 'longitude'}:
            df[col] = df[col].astype('str').apply(lambda x:x.strip('"')).astype('float')
    _convert_vogele_timestamps(df, "%d/%m/%Y %H:%M:%S")
    return title, df


if __name__ == '__main__':
    title, df = import_vogele_M119()
