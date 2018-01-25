# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 13:23:03 2018

@author: Dell
"""

import csv
import random
import numpy as np
from random import randint


reader = csv.reader(open("R.csv", "r"), delimiter=",")
x = list(reader)
result = np.array(x).astype("float")
R = np.matrix(result)

n = R.shape[0]
m = R.shape[1]      
k = 100 

def solve_iter( R, P, Q, u, v):

    alpha = 0.05
    lp = 0.05
    lq = 0.05
    n = R.shape[0]
    m = R.shape[1]      
    err = R[u,v] - np.dot(P[u,:],Q[:,v])
    
    list1 = [x - y for x, y in zip([err*x for x in Q[:,v]] , [lp*x for x in P[u,:]])]
    list2 = [x - y for x, y in zip([err*x for x in P[u,:]] , [lq*x for x in Q[:,v]])]
    
    Pu = np.array([x + y for x, y in zip(P[u,:] , [alpha*y for y in list1])][0])[0]
    Qv = np.array([x + y for x, y in zip(Q[:,v] , [alpha*y for y in list2])][0])[0]
    
    
    P_new = P
    Q_new = Q
    for i in range(0,k):
        Q_new[i,v] = Qv[i]
    for j in range(0,k):
        P_new[u,j] = Pu[j]

    return P_new, Q_new



#[[1,6,4,4],[11,2,3,4],[0,10,6,5],[9,-3,15,7]]
#R = np.matrix([[1,6,5.9,4],[11.5,1,2.2,1],[7,5,7,1],[-1,5,11,8]])

P = np.zeros(shape=(n,k))
Q = np.zeros(shape=(k,m))
P = np.matrix(np.random.random((n, k)))
Q = np.matrix(np.random.random((k, m)))


for t in range(0,5000):
    if t%100 == 0 :
        print(t)
    u = randint(0, n-1)
    v = randint(0, m-1)
    P, Q = solve_iter( R, P, Q, u, v)

#print(P)
#print(Q)

R_approx = P*Q

print(R_approx)



