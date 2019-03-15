# Deep Learning
from keras.models import Sequential
from keras.layers import LSTM, Dense, Conv2D, Dropout, MaxPooling2D, Reshape, Flatten, BatchNormalization
from keras.optimizers import Adam

def build_model(n_timesteps, n_features, n_outputs):
    
    # Optimiser
    opt = Adam(lr=0.001)
    
    # define model
    model = Sequential()
    model.add(Conv2D(9, (2,1), input_shape=(n_timesteps, n_features, 1)))
    model.add(MaxPooling2D(pool_size=(2,1)))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())
    
    model.add(Conv2D(9, (2,1), input_shape=(n_timesteps, n_features, 1)))
    model.add(MaxPooling2D(pool_size=(2,1)))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())
    
    model.add(Flatten())
    model.add(Dense(300, activation='sigmoid'))
    
    model.add(Dense(n_outputs))
    model.compile(loss='mse', optimizer=opt, metrics=['mae'])
    
    return model