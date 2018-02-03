# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 18:07:44 2018

@author: Dell
"""

#import findspark
#findspark.init()

from pyspark import SparkContext
import random
import numpy as np
from random import randint


sc = SparkContext.getOrCreate()
#[[1,6,4,4],[11,2,3,4],[0,10,6,5],[9,-3,15,7]]
#R = np.matrix([[1,6,5.9,4],[11.5,1,2.2,1],[7,5,7,1],[-1,5,11,8]])
#R = np.matrix([[1,1,5,1],[1,1,1,8],[12,1,1,1],[1,0,1,1]])
P = np.matrix([[0.1,0.2,0.3,0.4],[0.5,0.6,0.7,0.8],[0.9,1.0,1.1,1.2],[1.3,1.4,1.5,1.6]])
Q = np.matrix([[0,5,3,1],[1,0.1,0.1,0.1],[3,1,1.5,0.2],[0.4,0,1.5,1]])


reader = csv.reader(open("R.csv", "r"), delimiter=",")
x = list(reader)
result = np.array(x).astype("float")
R = np.matrix(result)

n = R.shape[0]
m = R.shape[1]      
k = 8
P = np.zeros(shape=(n,k))
Q = np.zeros(shape=(k,m))
P = np.matrix(np.random.random((n, k)))
Q = np.matrix(np.random.random((k, m)))
#P = np.matrix([[1,1,1,1],[1,1,1,1],[1,1,1,1],[1,1,1,1]])
#Q = np.matrix([[2,2,2,2],[2,2,2,2],[2,2,2,2],[2,2,2,2]])

#P = np.matrix([[0.4,0.1,0.6,0.4],[0,0,0,1.8],[0.9,1.0,1.1,1.2],[1.3,0,1.5,1.6]])
#Q = np.matrix([[0,5,0,0],[0.6,0.1,0,0.1],[1,1,2.5,0.2],[1.4,0.5,1.5,1.5]])

"""
for i in range(0,n):
    for j in range(0,k):
        P[i,j] = np.random.uniform(-1,1)
        
for i in range(0,k):
    for j in range(0,m):
        Q[i,j] = np.random.uniform(-1,1)
"""               


def wave_move(mat):
    n = mat.shape[0]
    m = mat.shape[1]
    mat_ret = np.zeros(shape=(n,m))
    for i in range(0,n):
        for j in range(0,m):
            if mat[i,j] == 1 and j != m-1 :
                mat_ret[i,j+1] = 1
            if mat[i,j] == 1 and j == m-1 :
                mat_ret[i,0] = 1
    return mat_ret
            
def blocs_decomposition(mat,u,v):
    final_matrix = []
    final_matrix = [[0 for x in range(v)] for y in range(u)] 

    n = mat.shape[0]
    m = mat.shape[1]    
    for i in range(0,u):
        for j in range(0,v):
            matrix = np.empty(shape=(int(n/u),int(m/v)))
            for k in range(0,int(n/u)):
                for l in range(0,int(m/v)):
                    matrix[k,l] = mat[k+i*int(n/u),l+j*int(m/v)]
                    
            final_matrix[i][j] = np.matrix(matrix)
        
    return final_matrix
           

def solve_iter( R, P, Q, u, v):

    alpha = 0.005
    lp = 0.001
    lq = 0.001
    n = R.shape[0]
    m = R.shape[1]      
    err = R[u,v] - np.dot(P[u,:],Q[:,v])
    
    list1 = [x - y for x, y in zip([err*x for x in Q[:,v]] , [lp*x for x in P[u,:]])]
    list2 = [x - y for x, y in zip([err*x for x in P[u,:]] , [lq*x for x in Q[:,v]])]
    
    Pu = np.array([x + y for x, y in zip(P[u,:] , [alpha*y for y in list1])])
    Qv = np.array([x + y for x, y in zip(Q[:,v] , [alpha*y for y in list2])])
    
    P_new = P
    Q_new = Q
    for i in range(0,k):
        Q_new[i,v] = Qv[i]
    for j in range(0,k):
        P_new[u,j] = Pu[j]

    return P_new, Q_new

def solve_iter_bis( R, P, Q, u, v, r, c, n, m):    
    P_new = P
    Q_new = Q
    for i in range(0,10000):
        u = randint(0,1)
        v = randint(0,1)
        P_new, Q_new = solve_iter( R, P_new , Q_new, u, v)

    P_new = complete_matrix_P(P_new, n, r)
    Q_new = complete_matrix_Q(Q_new, m, c)

    return [('P', P_new) , ('Q', Q_new)]

def id_matrix(n,m):
    mat = np.zeros(shape=(n,m))
    for i in range(0,min(n,m)):
        mat[i,i] = 1
    return mat

def sub_matrix_P( P, r1, r2):
    P_new = np.zeros(shape=(r2-r1,m))
    for i in range(r1,r2):
        for j in range(0,m):
            P_new[i-r1,j] = P[i,j]
    return P_new

def sub_matrix_Q( Q, c1, c2):
    Q_new = np.zeros(shape=(n,c2-c1))
    for i in range(0,n):
        for j in range(c1,c2):
            Q_new[i,j-c1] = Q[i,j]
    return Q_new

def complete_matrix_P( P, n, r):
    P_new = np.zeros(shape=(n,P.shape[1]))
    for i in range(r,r+P.shape[0]):
        for j in range(0,P.shape[1]):
            P_new[i,j] = P[i-r,j]
    return P_new

def complete_matrix_Q( Q, m, c):
    Q_new = np.zeros(shape=(Q.shape[0],m))
    for i in range(0,Q.shape[0]):
        for j in range(c,c+Q.shape[1]):
            Q_new[i,j] = Q[i,j-c]
    return Q_new


def distance( U , V ):
    n = U.shape[0]
    m = U.shape[1]
    dist = 0      
    for i in range(0,n):   
        for j in range(0,m):
            dist = dist + abs(U[i,j] - V[i,j])
    return dist
    



u = 2
v = 2
wave_matrix = id_matrix(u,v)
for t in range(0,100):
    print(t)

    R_approx = P.dot(Q)
    print("distance = ")
    d = distance(R_approx, R)
    print(d)
    
    blocs = blocs_decomposition(R,u,v)
    k = 0
    wave_blocs = [[0 for x in range(5)] for y in range(min(u,v))] 
    for i in range(0,u):
        for j in range(0,v):
            if wave_matrix[i,j] == 1 :
                wave_blocs[k][0] = blocs[i][j] #R
                wave_blocs[k][1] = sub_matrix_P( P, i*int(n/u), (i+1)*int(n/u)) #P
                wave_blocs[k][2] = sub_matrix_Q( Q, j*int(m/v), (j+1)*int(m/v)) #Q   
                wave_blocs[k][3] = i*int(n/u) #Q   
                wave_blocs[k][4] = j*int(m/v) #Q   
                
                k = k + 1
    
    #print(' -----------  wave_blocs  ------------- ')
    #print(wave_blocs)
    data = sc.parallelize(wave_blocs)
    result_map      = data.flatMap(lambda x: solve_iter_bis(x[0],x[1],x[2],randint(0, int(n/u)-1),randint(0, int(m/v)-1), x[3], x[4], n, m)) 
    result_reduce   = result_map.reduceByKey(lambda a,b: a+b)        
    
    P = result_reduce.collect()[1][1]
    Q = result_reduce.collect()[0][1]
    """
    print(result_reduce.collect())
    print(' ******* P and Q')
    print(P)
    print(Q)
    
    print('P*Q')
    print(R_approx)
    print('R')
    
    print(R)
    """
    
    wave_matrix = wave_move(wave_matrix)
    
print(P)
print(Q)

R_approx = P*Q

print(R_approx)



sc.stop()

