# -*- coding: utf-8 -*-
"""
Created on Sat Feb  3 01:17:42 2018

@author: Dell
"""



import random
import numpy as np
import csv
from random import randint

n = 8
m = 8
k = 8

mat = np.empty(shape=(n,m))
P = np.empty(shape=(n,k))
Q = np.empty(shape=(k,m))

for i in range(0,n):
    for j in range(0,k):
        P[i,j] = np.random.uniform(-1,1)

for i in range(0,k):
    for j in range(0,m):
        Q[i,j] = np.random.uniform(-1,1)

mat = P.dot(Q)

#print(mat)
#print(np.asarray(mat))

np.savetxt("R.csv", mat, delimiter=",", fmt='%.6f')
