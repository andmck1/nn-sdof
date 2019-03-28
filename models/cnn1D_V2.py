class Model():
    
    def __init__(self):
        self.mdl = None
        self.hist = None

    def create_model(self, input_shape):

        # Model
        self.mdl = Sequential()

        # -- INPUT
        self.mdl.add(Conv1D(9, (10), padding='same', input_shape=input_shape, activation='relu'))
        self.mdl.add(MaxPool1D(pool_size=(3)))
        self.mdl.add(Dropout(0.2))
        self.mdl.add(BatchNormalization())
        
        # -- HIDDEN
        self.mdl.add(Conv1D(9, (10), activation='relu'))
        self.mdl.add(MaxPool1D(pool_size=(3)))
        self.mdl.add(Dropout(0.2))
        self.mdl.add(BatchNormalization())

        self.mdl.add(Conv1D(9, (10), activation='relu'))
        self.mdl.add(MaxPool1D(pool_size=(3)))
        self.mdl.add(Dropout(0.2))
        self.mdl.add(BatchNormalization())
        
        self.mdl.add(Flatten())
        
        self.mdl.add(Dense(150, activation='sigmoid'))
        
        # -- OUTPUT
        self.mdl.add(Dense(n_output))


        # Prepare model for training
        self.mdl.compile(loss='mse',
                      optimizer='adam',
                      metrics=['mae'])

    
    def run_model(self, X_train, X_test, y_train, y_test):
    
        # Fit model
        self.hist = self.mdl.fit(X_train, y_train,
                            batch_size=5000,
                            epochs=300,
                            verbose=True,
                            validation_data=(X_test, y_test))