#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 31 11:39:08 2021

@author: kirineko
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor

sample_size = 100

X = np.linspace(0, 6, sample_size)
# y = np.sin(X)
y = X ** 2

for i in range(10):
    size = (i + 1) * 1000
    hidden_layer_sizes = (size,)
    model = MLPRegressor(hidden_layer_sizes=hidden_layer_sizes, activation='relu')
    model.fit(X.reshape((sample_size, 1)), y)
    y_pred = model.predict(X.reshape((sample_size, 1)))

    plt.show()
    plt.scatter(X, y)
    plt.scatter(X, y_pred)
    plt.title(str(hidden_layer_sizes))