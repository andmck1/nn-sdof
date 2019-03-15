__author__ = 'kr4in'
__date__ = '13-03-2019'
__description__ = 'Set of functions useful for evaluating time series data.'


# ---------------
# IMPORTS -------
# ---------------


# ---------------
# FUNCTIONS -----
# ---------------

def resids(mdl, x_test, y_test, n_input, n_output):
    """
    A function that returns residuals.

    Args:
        mdl (keras.model): Keras model that will predict new data.
        x_test (np.array): Input testing data.
        y_test (np.array): True values for evaluation.
        n_input (integer): Number of values in a sequence.

    Returns:
        np.array: Residuals (predicted - testing)

    """
    y_hats = mdl.predict(x_test)

    if y_hats.shape == y_test.shape:
        resids = (y_hats - y_test).reshape(-1, n_output)
        return resids
    else:
        print('Predicted shape: %s' % y_hats.shape)
        print('Evaluation shape: %s' % y_test.shape)
        print('Predictions and training data shape mismatch.')
