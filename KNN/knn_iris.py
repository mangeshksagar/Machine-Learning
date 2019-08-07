#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 00:02:46 2019

@author: mangesh.kshirsagar
"""

 #%% imort required packages
import numpy as np
import pandas as pd 
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import statistics

#%% load iris dataset    
iris = load_iris()
X = iris.data
y = iris.target

#%% split the dataset into train and test set
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.3)

#%% create distance functions 
def dist(train,test):
    return np.sqrt(sum((train - test)**2))

#%% Doule For loop for calculating dist of each test set with each train set 
distance = []
for i in np.arange(len(X_test)):
    for j in np.arange(len(X_train)):
        d = dist(X_test[i] , X_train[j])
        distance.append(d)

#%% reshape the distance into matrix
distance = np.array(distance)
distance.shape = (45,105)

#%%sort the distance and find the indices( n varies with Knn K and m varies for finding mode)
Acc_Knn =[]
K = [1,3,5]
for n in K:
    max = []
    for m in np.arange(len(y_test)):
        matrix =statistics.mode((y_train[np.argsort(distance)[:,0:n]])[m])
        max.append(matrix)
    Acc=accuracy_score(max,y_test)