import matplotlib.pyplot as plt
import seaborn as sns

from data import import_TF


def trim(df, column, threshold_temp, percentage_above):
    above_threshold = sum(df[column] > threshold_temp)
    above_threshold_pct = 100 * (above_threshold / len(df))
    if above_threshold_pct > percentage_above:
        return False
    else:
        del df[column]
        return True


def trim_data(df, threshold, percentage_above):
    for column_name in df.columns:
        if not trim(df, column_name, threshold, percentage_above):
            break

    for column_name in reversed(df.columns):
        if not trim(df, column_name, threshold, percentage_above):
            break
    return df


def estimate_road_edge_right(line, threshold):
    cond = line < threshold
    count = 0
    while True:
        if any(cond[count:count + 3]):
            count += 1
        else:
            break
    return count

def estimate_road_edge_left(line, threshold):
    cond = line < threshold
    count = len(line)
    while True:
        if any(cond[count - 3:count]):
            count -= 1
        else:
            break
    return count


def estimate_road_length(df, threshold):
    values = df.values
    for distance_idx in range(values.shape[0]):
        offset = estimate_road_edge_right(values[distance_idx, :], threshold)
        values[distance_idx, :offset] = 190

        offset = estimate_road_edge_left(values[distance_idx, :], threshold)
        values[distance_idx, offset:] = 190


def plot_data(df):
    snsplot = sns.heatmap(df, cmap='RdYlGn_r', vmin=60)#, square=True)
    return snsplot

if __name__ == '__main__':
    import config as cfg
    for n, df in enumerate(import_TF(cfg)):
        df = trim_data(df, cfg.trim_threshold, cfg.percentage_above)
        df = trim_data(df.T, cfg.trim_threshold, cfg.percentage_above).T
        estimate_road_length(df, cfg.roadlength_threshold)
        snsplot = plot_data(df)
        #plt.show()
        plt.savefig("trimmed_data{}.png".format(n), dpi=800)
