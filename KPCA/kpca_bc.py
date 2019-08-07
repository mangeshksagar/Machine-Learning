#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  6 19:00:59 2019

@author: mangesh.kshirsagar
"""

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from numpy import linalg as LA


X = load_breast_cancer().data
M = X.mean(axis=0)
nX = X - M
Z = np.transpose(nX)
y = load_breast_cancer().target

def polynomial(a1,a2):
    return ((1 + np.dot(a1,a2))**2)

k = polynomial(nX,Z)
mat=[]
for i in range(569):
    a=[]
    for j in range(569):
        a.append(1/569)
    mat.append(a)
L = np.array(mat)
KL = np.dot(k,L)
LK = np.dot(L,k)
LKL = np.dot(LK,L)

K = k - LK - KL + LKL

w,v = LA.eig(K)
real_eigen = w.real
real_eig_vec = v.real 
Lambda = np.cumsum(real_eigen) / sum(real_eigen)
#np.who(Lambda)
aa = np.array(Lambda)
N_lambda = aa[aa<=0.98]
N_lambda = N_lambda.tolist()
New_eig_vec = np.transpose(real_eig_vec)
New_eig_vec = real_eig_vec[0:np.argmax(N_lambda)+1,]
#T_new_eig_vec = np.transpose(New_eig_vec)
Mean = New_eig_vec.mean(axis=1)
Sigma = New_eig_vec.std(axis=1)
T_new_eig_vec = np.transpose(New_eig_vec)
std_eig_vec = (T_new_eig_vec - Mean) / Sigma

New_data = np.transpose(np.dot(np.transpose(std_eig_vec),k))

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(New_data, y, test_size=0.2, random_state=42)  

from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(random_state=1, max_depth=10)
model.fit(X_train,y_train)
y_pred = model.predict(X_test)

from sklearn.metrics import accuracy_score,f1_score,make_scorer
accuracy = accuracy_score(y_test,y_pred)

from sklearn.model_selection import cross_val_score
score = make_scorer(accuracy_score)
cv = cross_val_score(estimator=model,X=New_data,y=y,cv=5,scoring=score)
m_acc = np.max(cv)
print("Accuracy is ",accuracy)
print("CV Accuracy is ",m_acc)