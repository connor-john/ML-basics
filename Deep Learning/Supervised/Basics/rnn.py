# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 17:58:46 2020

@author: Connor
"""

# Recurrent Neural Network

### Data Preprocessing

# Importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Training set (Based on Google stock price of the last 5 years)
dataset_train = pd.read_csv('Google_Stock_Price_Train.csv')
training_set = dataset_train.iloc[:, 1:2].values

# Feature Scaling (Normalisation)
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0,1))
training_set_scaled = sc.fit_transform(training_set)

# Creating the data structure (60 timesteps and 1 output)
timesteps = 60
X_train = []
y_train = []
for i in range(timesteps, training_set_scaled.shape[0]):
    X_train.append(training_set_scaled[i - timesteps : i, 0])
    y_train.append(training_set_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)

# Reshaping
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

### Build RNN

# Importing Keras Libraries
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, Flatten

# Initialise RNN
model = Sequential()

# Adding first LSTM layer (incl. dropout)
model.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))
model.add(Dropout(0.2))

# Adding more layers
model.add(LSTM(units = 50, return_sequences = True))
model.add(Dropout(0.2))

model.add(LSTM(units = 50, return_sequences = True))
model.add(Dropout(0.2))

model.add(LSTM(units = 50, return_sequences = True))
model.add(Dropout(0.2))

model.add(Flatten())

# Output layer
model.add(Dense(units = 1))

# Compiling RNN
model.compile(optimizer = 'adam', loss = 'mean_squared_error')

# Fitting the RNN
model.fit(X_train, y_train, epochs = 100, batch_size = 32)

### Make Predictions

# Getting real results
dataset_test = pd.read_csv('Google_Stock_Price_Test.csv')
real_stock_prices = dataset_test.iloc[:, 1:2].values

# Getting predicted stock price
dataset_total = pd.concat((dataset_train['Open'], dataset_test['Open']), axis = 0)
inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60 : ].values
inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs)
X_test = []
for i in range(timesteps, 80):
    X_test.append(inputs[i - timesteps : i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
predicted_stock_price = model.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

### Visualise
plt.plot(real_stock_prices, color = 'red', label = 'Real Google Stock Price')
plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted Google Stock Price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()
