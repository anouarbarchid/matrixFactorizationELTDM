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
#R = np.matrix([[1,6,4,4],[11,2,3,4],[0,10,6,5],[9,-3,15,7]])
P = np.matrix([[0.1,0.2,0.3,0.4],[0.5,0.6,0.7,0.8],[0.9,1.0,1.1,1.2],[1.3,1.4,1.5,1.6]])
Q = np.matrix([[0,5,3,1],[1,0.1,0.1,0.1],[3,1,1.5,0.2],[0.4,0,1.5,1]])

R = P*Q
#R = np.matrix([[0.1,0.2,0.3,0.4],[0.5,0.6,0.7,0.8],[0.9,1.0,1.1,1.2],[1.3,1.4,1.5,1.6]])

n = R.shape[0]
m = R.shape[1]      
k = 2

P = np.zeros(shape=(n,k))
Q = np.zeros(shape=(k,m))
P = np.matrix(np.random.random((n, k)))
Q = np.matrix(np.random.random((k, m)))

for i in range(0,n):
    for j in range(0,k):
        P[i,j] = np.random.uniform(1,2)
        
for i in range(0,k):
    for j in range(0,m):
        Q[i,j] = np.random.uniform(1,2)
       
print(P)
print(Q)        
        
def solve_iter( R, P, Q, u, v):

    alpha = 0.05
    lp = 0.05
    lq = 0.05
    n = R.shape[0]
    m = R.shape[1]      
    err = R[u,v] - np.dot(P[u,:],Q[:,v])
    
    list1 = [x - y for x, y in zip([err*x for x in Q[:,v]] , [lp*x for x in P[u,:]])]
    list2 = [x - y for x, y in zip([err*x for x in P[u,:]] , [lq*x for x in Q[:,v]])]
    
    #print(list1)
    #print(list2)
    
    
    Pu = np.array([x + y for x, y in zip(P[u,:] , [alpha*y for y in list1])][0])[0]
    Qv = np.array([x + y for x, y in zip(Q[:,v] , [alpha*y for y in list2])][0])[0]
    
    
    P_new = P
    Q_new = Q
    for i in range(0,k):
        Q_new[i,v] = Qv[i]
    for j in range(0,k):
        P_new[u,j] = Pu[j]

    return P_new, Q_new


def distance( U , V ):
    n = R.shape[0]
    m = R.shape[1]
    dist = 0      
    for i in range(0,n):   
        for j in range(0,m):
            dist = dist + abs(U[i,j] - V[i,j])
    return dist
    



for t in range(0,100000):
    if t%100 == 0 :
        R_approx = P*Q
        print(t)
        print(distance(R,R_approx))
    u = randint(0, n-1)
    v = randint(0, m-1)
    P, Q = solve_iter( R, P, Q, u, v)

#print(P)
#print(Q)

R_approx = P*Q


print(R)
print(R_approx)



