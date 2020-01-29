# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 19:57:19 2020

@author: Connor
"""

# Artificial Neural Network

# Importing the libraries
import numpy as np
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13 ].values

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
transformer = ColumnTransformer([('one_hot_encoder', OneHotEncoder(), [1])], remainder = 'passthrough')
X = np.array(transformer.fit_transform(X), dtype=np.float)
X = X[:, 1:]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling (Very Important)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Setting up Keras
from keras.models import Sequential
from keras.layers import Dense

# Initialise ANN
classifier = Sequential()

# Adding Input layers and hidden layers
classifier.add(Dense(6, kernel_initializer='random_uniform', activation = 'relu', input_shape = (11,)))

# Add a hidden layer
classifier.add(Dense(6, kernel_initializer='random_uniform', activation = 'relu'))

# Add Output Layer
classifier.add(Dense(1, kernel_initializer='random_uniform', activation = 'sigmoid'))

# Compiling ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fit ANN to Training set
classifier.fit(X_train, y_train, batch_size = 10, epochs = 100)

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
