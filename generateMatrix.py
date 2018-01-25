# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 19:14:52 2018

@author: Dell
"""


import random
import numpy as np
import csv
from random import randint

n = 1024
m = 1024

mat = np.empty(shape=(n,m))

for i in range(0,n):
    for j in range(0,m):
        mat[i,j] = np.random.uniform(-1,1)


#print(mat)
#print(np.asarray(mat))

np.savetxt("R.csv", mat, delimiter=",", fmt='%.6f')



