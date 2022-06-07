#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  4 12:44:33 2021

@author: aysegulbarlas
"""

import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import tensorflow as tf

data = pd.read_csv("train.csv", header = 0, index_col = None)
test = pd.read_csv("test.csv")

X_train = np.array(data['Sequence'])
Y_train = np.array(data['Active'])
X_test = np.array(test['Sequence'])

alphabet = { 'R': 0, 'H': 1, 'K': 2, 'D': 3, 'E': 4, 'S': 5, 'T': 6, 'N': 7,
    'Q': 8,'C': 9, 'U': 10, 'G': 11, 'P': 12, 'A': 13, 'I':14, 'L':15, 'M': 16,
    'F': 17, 'W': 18, 'Y': 19, 'V': 20 }

onehot_arr = np.empty((len(X_train), 84))
onehot_arr_test = np.empty((len(X_test), 84))

for j, seq in enumerate(X_train):
    ind = [alphabet[ch] for ch in seq]
    one_hot = tf.one_hot(ind,21, dtype = tf.uint8)
    onehot_arr[j,:] = np.hstack(one_hot.numpy())
    
for j, seq in enumerate(X_test):
    ind = [alphabet[ch] for ch in seq]
    one_hot = tf.one_hot(ind,21, dtype = tf.uint8)
    onehot_arr_test[j,:] = np.hstack(one_hot.numpy())


svc = SVC(C = 11, class_weight='balanced', tol=1.9e-2)

svc.fit(onehot_arr, Y_train)
Y_pred = svc.predict(onehot_arr_test)

pd.DataFrame(Y_pred).to_csv("submit.csv", index= None, header= None)