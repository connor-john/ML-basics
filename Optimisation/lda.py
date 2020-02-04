# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 20:06:40 2020

@author: Connor
"""

# Linear Discriminant Analysis (LDA)

# Importing the libraries
import pandas as pd

# Importing the dataset (Using UCI wine DataSet)
dataset = pd.read_csv('Wine.csv')
X = dataset.iloc[:, 0 : -1].values
y = dataset.iloc[:, -1].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Applying LDA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
lda = LDA(n_components = 2)
X_train = lda.fit_transform(X_train, y_train)
X_test = lda.transform(X_test)
