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
from keras.layers import Dropout

# Initialise ANN
classifier = Sequential()

# Adding Input layers and hidden layers (with Dropout)
classifier.add(Dense(6, kernel_initializer='random_uniform', activation = 'relu', input_shape = (11,)))
classifier.add(Dropout(p = 0.1))

# Add a hidden layer
classifier.add(Dense(6, kernel_initializer='random_uniform', activation = 'relu'))
classifier.add(Dropout(p = 0.1))

# Add Output Layer
classifier.add(Dense(1, kernel_initializer='random_uniform', activation = 'sigmoid'))

# Compiling ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fit ANN to Training set
classifier.fit(X_train, y_train, batch_size = 10, epochs = 100)

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

# Taking in a new input prediction
new_prediction = classifier.predict(sc.transform(np.array([[0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]])))
new_prediction = (new_prediction > 0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# Evaluating the model
from keras.wrappers.scikit_learn import KerasClassifier 
from sklearn.model_selection import cross_val_score
def build_classifier():
    classifier = Sequential()
    classifier.add(Dense(6, kernel_initializer='random_uniform', activation = 'relu', input_shape = (11,)))
    classifier.add(Dense(6, kernel_initializer='random_uniform', activation = 'relu'))
    classifier.add(Dense(1, kernel_initializer='random_uniform', activation = 'sigmoid'))
    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier

classifier = KerasClassifier(build_fn = build_classifier, batch_size = 10, epochs = 100)
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10, n_jobs = 1)
mean = accuracies.mean()
variance = accuracies.std()

# Tuning the model
from keras.wrappers.scikit_learn import KerasClassifier 
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense
def build_classifier(optimizer):
    classifier = Sequential()
    classifier.add(Dense(6, kernel_initializer='random_uniform', activation = 'relu', input_shape = (11,)))
    classifier.add(Dense(6, kernel_initializer='random_uniform', activation = 'relu'))
    classifier.add(Dense(1, kernel_initializer='random_uniform', activation = 'sigmoid'))
    classifier.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier

classifier = KerasClassifier(build_fn = build_classifier)
parameters = {'batch_size': [25, 32],
              'epochs': [100, 500],
              'optimizer': ['adam', 'rmsprop']}
grid_search = GridSearchCV(estimator = classifier, param_grid = parameters, scoring = 'accuracy', cv = 10)
grid_search = grid_search.fit(X_train, y_train)
best_param = grid_search.best_params_
best_acc = grid_search.best_score_



















