# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 20:39:39 2020

@author: Connor
@Desc: Combining SOM with ANN to predict Frauds
"""

# Hybrid Deep Learning Model

### Building SOM

# Importing the libraries
import numpy as np
import pandas as pd

# Importing the dataset (Using UCI Machine learning Repository)
dataset = pd.read_csv('Credit_Card_Applications.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
X = sc.fit_transform(X)

# Training the SOM (Using minisom https://github.com/JustGlowing/minisom/blob/master/minisom.py)
from minisom import MiniSom
som = MiniSom(x = 10, y = 10, input_len = 15, sigma = 1.0, learning_rate = 0.5)
som.random_weights_init(X)
som.train_random(data = X, num_iteration = 100)

# Visualizing the results
from pylab import bone, pcolor, colorbar, plot, show
bone()
pcolor(som.distance_map().T)
colorbar()
markers = ['o', 's']
colors = ['r', 'g']
for i, x in enumerate(X):
    w = som.winner(x)
    plot(w[0] + 0.5,
         w[1] + 0.5,
         markers[y[i]],
         markeredgecolor = colors[y[i]],
         markerfacecolor = 'None',
         markersize = 10,
         markeredgewidth = 2)
show()

# Finding the frauds (dependent on the result of the SOM)
mappings = som.win_map(X)
frauds = np.concatenate((mappings[(7,7)], mappings[(3,5)]), axis = 0)
frauds = sc.inverse_transform(frauds)

### Preparing Supervised Model

# Creating Matrix
customers = dataset.iloc[:, 1:].values

# Creating dependent variable
is_fraud = np.zeros(len(dataset))
for i in range(len(dataset)):
    if dataset.iloc[i, 0] in frauds:
        is_fraud[i] = 1

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
customers = sc.fit_transform(customers)

### Creating the ANN

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense

# Initialise ANN
classifier = Sequential()

# Adding Input layers and hidden layers (with Dropout)
classifier.add(Dense(2, kernel_initializer='random_uniform', activation = 'relu', input_shape = (15,)))

# Add Output Layer
classifier.add(Dense(1, kernel_initializer='random_uniform', activation = 'sigmoid'))

# Compiling ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fit ANN to Training set
classifier.fit(customers, is_fraud, batch_size = 1, epochs = 2)

# Predicting the probabilities of frauds
y_pred = classifier.predict(customers)
y_pred = np.concatenate((dataset.iloc[:,0:1].values, y_pred), axis = 1)
y_pred = y_pred[y_pred[:, 1].argsort()]




