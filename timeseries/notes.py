def walk_forward(df, x_labels, y_labels, n_input, n_output, n_input_labels, split_per=None, split_data=False):
    td_shape = df.shape
    td_l = td_shape[0]
    
    x_indices = [range(i, i+n_input) for i in range(td_l-n_input-n_output)]
    y_indices = [range(i, i+n_output) for i in range(n_input, td_l-n_output)]
    
    x = df.loc[:, x_labels].values[x_indices]
    y = df.loc[:, y_labels].values[y_indices]
    
    if split_data:
        split_index = int(len(x)*split_per)
        x = x.reshape(-1, n_input, n_input_labels, 1)
        y = y.reshape(-1, n_output)
        X_train, X_test, y_train, y_test = (x[:split_index], x[split_index:], y[:split_index], y[split_index:])
        return X_train, X_test, y_train, y_test
    
    return x, y
