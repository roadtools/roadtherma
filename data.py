import pandas as pd

import config as cfg

def import_TF():
    for file_ in cfg.TF_M14_files:
        temperatures = ['T{}'.format(n) for n in range(141)]
        columns = ['distance'] + temperatures + ['distance_again']
        df = pd.read_csv(file_, skiprows=7, delimiter=',', names=columns)
        del df['distance_again']
        del df['T140'] # This is the last column of the dataset which is empty
        yield df


temperatures_voegele = ['T{}'.format(n) for n in range(52)]
VOEGELE_BASE_COLUMNS = ['time', 'distance', 'latitude', 'longitude']

def import_vogele_example():
    """ Old example code. This is probably not going to be used. """
    columns = VOEGELE_BASE_COLUMNS + temperatures_voegele
    df = pd.read_csv(cfg.voegele_example, skiprows=2, delimiter=';', names=columns)
    df.temperatures = temperatures_voegele
    return df


def import_vogele_data():
    """
    Import real voegele data.
    NOTE: The formatting of the data has changed from the example data above
    """
    import csv
    # NOTE Removed last line from the original data-file since it only contained 'No data to display'
    columns = VOEGELE_BASE_COLUMNS + ['sginal_quality'] + temperatures_voegele
    df = pd.read_csv(cfg.voegele_taulov, skiprows=3, delimiter=',', names=columns, quoting=csv.QUOTE_NONE, quotechar='"', doublequote=True)
    df.temperatures = temperatures_voegele
    for col in df.columns:
        if col == 'time':
            df[col] = df[col].apply(lambda x:x.strip('"'))
        if col in set(temperatures_voegele) | {'distance', 'latitude', 'longitude'}:
            df[col] = df[col].astype('str').apply(lambda x:x.strip('"')).astype('float')
    return df


if __name__ == '__main__':
    df = import_vogele_data()
