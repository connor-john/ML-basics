# -*- coding: utf-8 -*-
"""
Created on Fri Jan 31 21:10:53 2020

@author: Connor
"""

# Principal Component Analysis (PCA)

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

# Applying PCA
from sklearn.decomposition import PCA
pca = PCA(n_components = 2)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)
variance = pca.explained_variance_ratio_
