#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 27 16:10:03 2021

@author: kirineko
"""

import numpy as np
import matplotlib.pyplot as plt

#%%
print('hello world')
print('123')

#%%
a = np.arange(10)
print(a)

#%%
school = 'fudan university'
print(school.title())

#%%

arr_lin = np.linspace(0, 1, 101)
arr_lin2 = np.linspace(1, 2, 101)

arr_lin4 = arr_lin * arr_lin - arr_lin
plt.plot(arr_lin4)

#%%

arr_sq = np.zeros(100)

for i in range(100):
    arr_sq[i] = i * i
    
# %%
    
arr_flag = [0 if num < 50 else 1 for num in arr_sq]

# %%

arr_flag = np.zeros(100)

for i in range(100):
    if arr_sq[i] < 50:
        arr_flag[i] = 0
    else:
        arr_flag[i] = 1
        
#%%

def f(x):
    y = x**2 + np.sin(x) - np.exp(x)
    return y

a = np.linspace(-1, 1, 100)
b = f(a)

# plt.plot(a, b)
plt.scatter(a, b)
