#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  7 15:31:56 2019

@author: mangesh.kshirsagar
"""

"""Kernel K-nearest neighbor algorithm"""

"Import Libraries"
import numpy as np
import pandas as pd
from sklearn import preprocessing,datasets

"Import dataset"

"data = pd.DataFrame(datasets.load_iris().data)"
data = pd.DataFrame(datasets.load_breast_cancer().data)
#data = pd.read_csv("breast_cancer.csv",sep=",")
#data.drop(['id'],axis = 1,inplace=True)
y = datasets.load_breast_cancer().target

"Scale input data"
scaleData = pd.DataFrame(preprocessing.scale(data))

#X = data.drop(['diagnosis'],axis = 1)
#y = np.array(data['diagnosis'].tolist())
"Different Kernels"
A = 1
B = 1

def polynomial_kernel(x1,x2,p = 3):
    return (A + B * np.dot(x1,x2))**p
    
def linear_kernel(x1,x2):
    return np.dot(x1,x2)

def gaussian_kernel(x1,x2,sigma = 5.0):
    return np.exp(-np.linalg.norm(x1-x2)**2/(2 * sigma**2))

def distance(x1,x2,obj = polynomial_kernel):
    Z1 = obj(x1,x1)
    Z2 = obj(x1,x2)
    Z3 = obj(x2,x2)
    return np.sqrt(Z1 - 2 * Z2 + Z3)

#X_train,X_test,y_train,y_test = cross_validation.train_test_split(X,y)   

"Shuffle data"
shuf = np.random.choice(5,len(scaleData),replace=True)

"define folds and k neighbors"
fold = 5
K = np.arange(1,np.round(np.sqrt(len(data))),2)
fold_accuracy = []

"Loop over neighbors"
for nbr in K:
    "Loop over folds"
    for i in np.arange(fold):
        "divide data into test and train"
        X_test = scaleData.iloc[np.where(shuf == i)]
        X_train = scaleData.iloc[np.where(shuf != i)]
        y_test = y[np.where(shuf == i)]
        y_train = y[np.where(shuf != i)]
        y_pred = []
        
        "loops for calculating distance of each example in test with train"
        for j in np.arange(len(X_test)):
            dist = []
            
            for k in np.arange(len(X_train)):
                d = distance(X_test.iloc[j],X_train.iloc[k])
                dist.append(d)
            
            "min k neighbors"
            min_values = sorted(dist)[0:int(nbr)]
            "min k neighbors index"
            min_index = np.where(pd.DataFrame(dist).isin(min_values))[0]
            classes = y_train[min_index]
            
            "count maximum number of neighbors for each example in test"
            y_pred.append(np.argmax(pd.Series(classes).value_counts()))
            
        "accuracy "
        acc = (sum(y_pred == y_test) / len(y_test))
        #print(acc)
        fold_accuracy.append(acc)
    "mean fold accuracy for each neighbors"
    accuracy = np.mean(fold_accuracy)
    print("fold accuracy for %d neighbours is %f"%(nbr,accuracy))