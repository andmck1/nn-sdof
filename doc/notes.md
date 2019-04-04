# Notes
## 04/03/19
- Instead of using given data I am now generating my own. Looks a little
different to Sam's but similar and stable.
- Ran model on no noise data and got R2=0.9082 MAE=0.0092 MSE=1.4365E-4
  - Ran for 5 epochs, batch-size=32. Was slow.
  - Running with a faster model. No noise just now so should be able to do
  extremely well. The walk-forward had 250 points worth of previous data (500
  total), but looking at the equations you only need 1.
 - Model was:

```
# Optimiser
opt = Adam(lr=0.01)

# define model
self.mdl = Sequential()

self.mdl.add(Conv2D(9, (2,3), input_shape=input_shape))
self.mdl.add(MaxPooling2D(pool_size=(2,1)))
self.mdl.add(Dropout(0.2))
self.mdl.add(BatchNormalization())

self.mdl.add(Conv2D(9, (2,1)))
self.mdl.add(MaxPooling2D(pool_size=(2,1)))
self.mdl.add(Dropout(0.2))
self.mdl.add(BatchNormalization())

self.mdl.add(Conv2D(9, (2,1)))
self.mdl.add(Dropout(0.2))
self.mdl.add(BatchNormalization())

self.mdl.add(Flatten())
self.mdl.add(Dense(300, activation='sigmoid'))

self.mdl.add(Dense(n_outputs))

# Prepare model for training
self.mdl.compile(loss='mae',
              optimizer=opt,
              metrics=['mse'])
```

- Ran again with 1 previous data point predicting the next 99 points.
 - Got R2=0.9982 MAE=0.0018 MSE=5.2761E-6
 - Ran for 100 epochs, batch_size=100. V quick (fewer data, fewer parameters).
 - Pretty successful this time. Decrease in MSE indicates fewer outlier,
 decrease in MAE suggests closer fit to data, increase in R2 suggests the
 variance is almost entirely explained. The residual mean is
 1.0704E-3 +/- 2.0323E-3 with a skewness of 0.1250, indicating that there is a
 slight bias on the left side of the tail and that it tends to over predict.
 - Ran with model:

```
class Model():

    def __init__(self):
        self.mdl = None
        self.hist = None

    def create_model(self, input_shape, n_outputs):

        # Optimiser
        opt = Adam(lr=0.001)

        # define model
        self.mdl = Sequential()

        self.mdl.add(Conv2D(9, (1,3), input_shape=input_shape))
        self.mdl.add(MaxPooling2D(pool_size=(2,1)))
        self.mdl.add(Dropout(0.2))
        self.mdl.add(BatchNormalization())

        self.mdl.add(Flatten())
        self.mdl.add(Dense(300, activation='sigmoid'))
        self.mdl.add(Dense(300, activation='sigmoid'))

        self.mdl.add(Dense(n_outputs))

        # Prepare model for training
        self.mdl.compile(loss='mae',
                      optimizer=opt,
                      metrics=['mse'])


    def run_model(self, X_train, X_test, y_train, y_test, callback):

        # Fit model
        self.hist = self.mdl.fit(X_train, y_train,
                                batch_size=1000,
                                epochs=100,
                                verbose=True,
                                validation_data=(X_test, y_test),
                                callbacks=[callback])
```

- Interesting it did so well with DNN - makes sense based on the Finite-
Difference approach.
 - Removing CNN completely.
 - Meh. Didn't work as well. Resids std was 4.7E-3.
- One of the cases in the paper is predict the acceleration from the excitation
force only. Reckon you can calculate the displacement from the excitaiton but
lets start easy.
  - R2 0.8012, resids mean -1.08E-3 +/- 2.13E-2.
  - Interestingly there is a consistent sinusoidal error in the residuals.
  - Is the model just staying close to the mean and not finding the average
  movement? A kind of overfit? Might try adding extra convolutional layer.
  - Model:

```
class Model():

    def __init__(self):
        self.mdl = None
        self.hist = None

    def create_model(self, input_shape, n_outputs):

        # Optimiser
        opt = Adam(lr=0.001)

        # define model
        self.mdl = Sequential()

        self.mdl.add(Conv2D(9, (1,1), input_shape=input_shape))
        self.mdl.add(Dropout(0.2))
        self.mdl.add(BatchNormalization())

        self.mdl.add(Flatten())
        self.mdl.add(Dense(300, activation='sigmoid'))

        self.mdl.add(Dense(n_outputs))

        # Prepare model for training
        self.mdl.compile(loss='mae',
                      optimizer=opt,
                      metrics=['mse'])


    def run_model(self, X_train, X_test, y_train, y_test, callback):

        # Fit model
        self.hist = self.mdl.fit(X_train, y_train,
                                batch_size=1000,
                                epochs=100,
                                verbose=True,
                                validation_data=(X_test, y_test),
                                callbacks=[callback])
```

- Extra convolutional layer helped remove residuals bias but the data doesn't
fit at all now. Tbh the convolutional layer with a kernel size of 1x1 doesn't
really make sense....thinking about it, you should take in this point and the
point before to predict this point's acceleration. Onward! Kernel size of 2x1
- Epochs might need to increase as well. Error is still decreasing when it
stops
- Added dropout and BatchNormalization on dense hidden layer. No effect.
Rekon it's underfitting. Considering it might need a longer dependancy as only
giving it....wait how would be able to work out the acceleration given the
excitation only....it would need to know the previous acceleration as well.
Need to input the previous seen acceleration points, but obviously not give
it what it's trying to predict.
- DOIYAHHHH. Yeah that's decent now. R2=0.9702, MAE=0.0065, MSE=6.7847E-5,
resids mean = 1.5557E-4 +/- 8.2355E-3, resids skew = -0.0374. Resids suggest
some underlying sinusoidal bias still present. Possible overfit.
 - This is kind of cheating...in real life would you have the real
 acceleration? Maybe....not sure. I reckon it's okay...ideally I want to create
 a situation where it just progressively takes more excitation data and tests.
 That's the RNN boi though.
 - Also I now need to test with noise. Gonna add some normally distributed
 noise.
- Added 10% noise (finding acc with exc example) to acceleration. Not sure
how the boy determined what "10%" was so I just made it 10% of the standard
deviation.
- Comparable results (R2=0.9620)...suspiciously good. Checking with 50%.
- R2 with 50% noise was ~0.75. Siiiiick. Reckon the conv net is done now.
Onto adding the RNN!!.
- Model:

```
class Model():

    def __init__(self):
        self.mdl = None
        self.hist = None

    def create_model(self, input_shape, n_outputs):

        # Optimiser
        opt = Adam(lr=0.001)

        # define model
        self.mdl = Sequential()

        self.mdl.add(Conv2D(9, (2,1), input_shape=input_shape))
        self.mdl.add(Dropout(0.2))
        self.mdl.add(BatchNormalization())

        self.mdl.add(Flatten())
        self.mdl.add(Dense(500, activation='relu'))
        self.mdl.add(Dropout(0.2))
        self.mdl.add(BatchNormalization())

        self.mdl.add(Dense(n_outputs))

        # Prepare model for training
        self.mdl.compile(loss='mae',
                      optimizer=opt,
                      metrics=['mse'])


    def run_model(self, X_train, X_test, y_train, y_test, callback):

        # Fit model
        self.hist = self.mdl.fit(X_train, y_train,
                                batch_size=1000,
                                epochs=100,
                                verbose=True,
                                validation_data=(X_test, y_test),
                                callbacks=[callback])
```

---
