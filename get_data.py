# File
import numpy as np
import pandas as pd


def extract_and_save_to_df():
    # Read in data from .zips
    print('Reading acc')
    acc_raw = np.genfromtxt('./data/acc.csv', delimiter=',')
    print('Reading vel')
    vel_raw = np.genfromtxt('./data/vel.csv', delimiter=',')
    print('Reading exc')
    exc_raw = np.genfromtxt('./data/exc.csv', delimiter=',')
    print('Reading disp')
    disp_raw = np.genfromtxt('./data/disp.csv', delimiter=',')

    # Get different features as dataframes and concatenate
    print('Arranging data as dataframe...')
    acc_df = pd.DataFrame(acc_raw.T, columns=['acc', 'acc_10noise'])
    vel_df = pd.DataFrame(vel_raw.T, columns=['vel', 'vel_10noise'])
    exc_df = pd.DataFrame(exc_raw.T, columns=['exc', 'exc_10noise'])
    disp_df = pd.DataFrame(disp_raw.T, columns=['disp', 'disp_10noise'])
    data = pd.concat([acc_df, vel_df, exc_df, disp_df], axis=1)

    # Get different datasets
    print('Getting noise and no noise data')
    no_noise_data = data.loc[:, [x for x in data.columns if 'noise' not in x]]
    noise_data = data.loc[:, [x for x in data.columns if 'noise' in x]]

    # Scale them both
    print('Scaling data...')
    input_scaled = pd.DataFrame()
    for col in no_noise_data.columns:
        mean = no_noise_data[col].mean()
        minimum = no_noise_data[col].min()
        maximum = no_noise_data[col].max()
        input_scaled[col] = (no_noise_data[col]-mean)/(maximum - minimum)
        input_scaled[col] = input_scaled[col] / 2.0 + 0.5

    noise_input_scaled = pd.DataFrame()
    for col in noise_data.columns:
        mean = noise_data[col].mean()
        minimum = noise_data[col].min()
        maximum = noise_data[col].max()
        noise_input_scaled[col] = (noise_data[col]-mean)/(maximum - minimum)
        noise_input_scaled[col] = noise_input_scaled[col] / 2.0 + 0.5

    # Save to pickle
    print('Saving dataframes to pickles...')
    input_scaled.to_pickle('./data/train_df.pkl')
    noise_input_scaled.to_pickle('./data/10per_noise_train_df.pkl')

    print('Finished!')


def newmann_beta(m, c, k, dt, f, number_of_time_points):

    # Initialise
    x = np.zeros(number_of_time_points)
    v = np.zeros(number_of_time_points)
    a = np.zeros(number_of_time_points)

    # Initial Conditions
    x[0] = 0
    v[0] = 0
    a[0] = f[0] / m

    for i in range(1, number_of_time_points):
        df = f[i] - f[i-1]
        kk = ((6.0 * m) / (dt**2)) + ((3.0 * c) / dt) + k
        ff = df + ((3.0 * m) + (dt / 2.0 * c)) * a[i-1] + ((6.0 / dt * m) + (3.0 * c)) * v[i-1]
        dx = ff / kk

        x[i] = dx + x[i-1]
        v[i] = (-2.0 * v[i-1]) - (dt / 2.0 * a[i-1]) + (3.0 / dt * dx)
        a[i] = (-1.0 / m) * ((c * v[i]) + (k * x[i]) - f[i])

    cols = ['exc', 'dis', 'vel', 'acc']
    data = dict(zip(cols, [f, x, v, a]))
    df = pd.DataFrame(data)

    return df
