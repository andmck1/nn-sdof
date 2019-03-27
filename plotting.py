import matplotlib.pyplot as plt
import seaborn as sns


def plot_stacked_timesseries(data, top_col, bottom_col,
                             quantity, top_y_label, bottom_y_label):
    sns.set(style="ticks", color_codes=True)
    plt.subplot(2, 1, 1)
    plt.plot(data.index, data.loc[:, top_col], '.-')
    plt.title(quantity + ' over Time')
    plt.ylabel(top_y_label)

    plt.subplot(2, 1, 2)
    plt.plot(data.index, data.loc[:, bottom_col], '.-')
    plt.xlabel('time (units)')
    plt.ylabel(bottom_y_label)
    plt.show()
