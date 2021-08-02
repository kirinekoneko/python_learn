#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 30 10:51:03 2021

@author: kirineko
"""

import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

path = './iris.csv'
data = pd.read_csv(path, index_col=0)


coloums = data.columns
data_v = data.values

data.iloc[0, 0]
data.iloc[1, :]
data.iloc[:, -1]

model = KNeighborsClassifier(n_neighbors=5, p=1)
X = data.iloc[:, 0:4]
y = data.iloc[:, -1]

model.fit(X, y)
model.score(X, y)

area1 = data.iloc[:, 0] * data.iloc[:, 1]
area2 = data.iloc[:, 2] * data.iloc[:, 3]
data['area1'] = area1
data['area2'] = area2
data_final = data.iloc[:, [0, 1, 2, 3, 4, 5, 6]]

save_path = './iris_new.csv'
data_final.to_csv(save_path)

#%%

from sklearn.model_selection import train_test_split

train_X, test_X, train_y, test_y = train_test_split(X, y, test_size = 0.4)

model = KNeighborsClassifier(n_neighbors=5)
model.fit(train_X, train_y)

print(model.score(train_X, train_y))
print(model.score(test_X, test_y))

#%%

import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return x * x - 2 * x - 5


def fderiv(x):
    return 2 * x - 2


learning_rate = 0.1
n_iter = 100

xs = np.zeros(n_iter + 1)
xs[0] = 100

for i in range(n_iter):
    xs[i + 1] = xs[i] - learning_rate * fderiv(xs[i])

plt.plot(xs)

#%%

from scipy.optimize import minimize

a = minimize(f, x0=100).x

def f2(x):
    return np.exp(-x ** 2) * (x**2)

print(f2(1))
a = minimize(f2, x0=5).x
b = minimize(f2, x0=-5).x

x = np.linspace(-5, 5)
y = f2(x)
plt.plot(x, y)

