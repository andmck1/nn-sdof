__author__ = 'kr4in'
__date__ = '13-03-2019'
__description__ = 'Set of functions useful for preprocessing time series data.'


# ---------------
# IMPORTS -------
# ---------------
import numpy as np
from keras.preprocessing.sequence import pad_sequences


# ---------------
# FUNCTIONS -----
# ---------------

def to_supervised(train_data, n_input, n_output, walk_forward=False):
    """
    Create walk-forward training data from passed data and the number of points
    in each sequence value.

    Args:
        train_data (pd.DataFrame): 2-d time-series data (rows=timesteps,
        columns=features)

        n_input (integer): number of points in each input sequence value (eg.
        if breaking down into weeks then n_input=7).

        n_output (integer): number of points in each output sequence value.

    Returns:
        (np.array, np.array): return a tuple of two numpy arrays where array 1
        is the training data and array 2 is the testing data.

    """
    # Inferred Parameters
    td_shape = train_data.shape
    td_l = td_shape[0]

    # Indices for getting data
    if walk_forward:
        x_indices = [range(0, i+n_input) for i in range(td_l-n_input-n_output)]
        y_indices = [range(n_input, i+n_output) for i in range(n_input, td_l-n_output)]

        x = pad_sequences(np.array([np.array(train_data[idxs]) for idxs in x_indices]))
        y = pad_sequences(np.array([np.array(train_data[idxs]) for idxs in y_indices]))
    else:
        x_indices = [range(i, i+n_input) for i in range(td_l-n_input-n_output)]
        y_indices = [range(i, i+n_output) for i in range(n_input, td_l-n_output)]

        x = train_data[x_indices]
        y = train_data[y_indices]

    return x, y
