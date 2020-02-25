# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 20:07:13 2020

@author: Connor
"""

# Self Organising Map (SOM)

# Import libraries
import numpy as np
import pandas as pd

# Importing the dataset (Using UCI Machine learning Repository)
dataset = pd.read_csv("Credit_Card_Applications.csv")
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:, -1].values 

# Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
X = sc.fit_transform(X)

# Training the SOM (Using minisom https://github.com/JustGlowing/minisom/blob/master/minisom.py)
from minisom import MiniSom
sigma = 1.0
learning_rate = 0.5
som = MiniSom(x = 10, y = 10, input_len = np.size(X, 1), sigma = sigma, learning_rate = learning_rate)
som.random_weights_init(X)
som.train_random(data = X, num_iteration = 100)

# Visualising results
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

# Finding Frauds
mappings = som.win_map(X)
frauds = mappings[(8, 2)]
frauds = sc.inverse_transform(frauds)