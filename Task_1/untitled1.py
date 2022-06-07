#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 17 19:14:32 2021

@author: aysegulbarlas
"""


import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error as mse
import numpy as np


data = pd.read_csv("train.csv")

X_train = np.array(data[['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9',
                        'x10', 'x11', 'x12', 'x13']])

Y_train = np.array(data[['y']])
    

alphas = [0.1, 1, 10, 100, 200]
    
output = []

  
for a in alphas:
    sumrmse = 0
    regressor = Ridge(alpha = a, solver = 'cholesky')
    kf = KFold(n_splits = 10, shuffle = True)
    for train_index, test_index in kf.split(X_train):
        regressor.fit(X_train[train_index], Y_train[train_index])
        Y_pred = regressor.predict(X_train[test_index])
        sumrmse += np.sqrt(mse(Y_train[test_index], Y_pred))
    avg = sumrmse / 10
    output.append(avg)
           
np.savetxt('output1a.csv', output, delimiter = ',')
    
    
        
    