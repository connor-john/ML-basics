# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 19:42:37 2020

@author: Connor
"""

# Apriori

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset and Preparing data
dataset = pd.read_csv('Market_Basket_Optimisation.csv', header = None)
transactions = []
for i in range(0, len(dataset.index)):
    transactions.append([str(dataset.values[i, j]) for j in range(0, len(dataset.columns))])

# Training Apriori on the dataset
from apyori import apriori
rules = apriori(transactions, min_support = 0.003, min_confidence = 0.2, min_lift = 3, min_length = 2)

# Visiualising the Results
AR = list(rules)
result = [list(AR[i][0]) for i in range(0, len(AR))]