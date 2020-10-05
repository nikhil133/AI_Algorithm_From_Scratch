#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug  3 14:17:27 2019

@author: Nikhil
"""

import numpy as np

import matplotlib.pyplot as plt
from sklearn.datasets.samples_generator import make_blobs
#creating random dataset
(X,Y) = make_blobs(n_samples=100,n_features=2,centers=2,cluster_std=1.05,random_state=20)
n= Y.shape[0]
one=np.ones(n)
X1=np.c_[one,X]
print(X1)
plt.scatter(X1[:,1],X1[:,2],marker='o',c=Y)
plt.show()

data=np.c_[X,Y]
print(np.corrcoef([X[:,0],X[:,1],Y]))

theta = np.zeros(3)
learning_rate=0.001

def sigmoid(h):
    return 1.0/(1.0+np.exp(-1.0*h))

def gradientDescent(X,Y,theta,alpha):
    cost_list=[]
    itter=[]
    cost_list.append(float('inf'))
    run =True
    i=0
    while run:
        h=np.dot(X,theta)
        predicted=sigmoid(h)
        error=(-Y*np.log(predicted))+((1-Y)*np.log(1-predicted))
        cost=((np.dot(error,error.transpose()))/n)*alpha
        cost_list.append(cost)
        theta = theta+((1/n)*X.transpose().dot(error))*alpha
        if cost_list[i]-cost_list[i+1]<1e-9:# if cost is not decreasing at difference greater than 1e-9 that is negligble then terminate
            run=False
        i=i+1 
        itter.append(i)
    cost_list.pop(0)
    return cost_list,theta,itter

cost,m,itter=gradientDescent(X1,Y,theta,learning_rate)    

plt.plot(itter,cost)
plt.show()

def predict(X1,m):
    predict_list=[]
    for x in X1:
        h=x.dot(m)
        predict_list.append(sigmoid(h))
    return predict_list    
        
predicted=predict(X1,m)

for i,v in enumerate(predicted):
    if v>=0.5:
        predicted[i]=1
    else:
        predicted[i]=0

error=sum((predicted-Y)**2)
print('error:',error)
accuracy=1-(error/100)
print('Accuracy :',accuracy)
        
        
plt.scatter(X1[:,1],X1[:,2],marker="^",c=predicted)
plt.show()

plt.show()
