#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 28 22:00:10 2019

@author: mangesh.kshirsagar
"""

import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.datasets import load_breast_cancer 
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import classification_report, confusion_matrix

# Load data
cancer = load_breast_cancer()
df = pd.DataFrame(np.c_ [cancer['data'],cancer['target']], columns=np.append (cancer['feature_names'],['target']))

# Seperating data
X = df.drop(['target'], axis = 1)
y = df['target']

# Splitting data into Train & Test sets
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=42)

# Creating SVM Classifier
SVC_model = SVC()
SVC_model.fit(X_train,y_train)
y_predict = SVC_model.predict(X_test)
cm = confusion_matrix(y_test, y_predict)

# Data normalization
scaler = MinMaxScaler()
x_train_Scaled = scaler.fit_transform(X_train)
x_test_Scaled = scaler.fit_transform(X_test)

# SVM on Scaled Data
SVC_model.fit(x_train_Scaled, y_train)
y_predict = SVC_model.predict(x_test_Scaled)
cm = confusion_matrix(y_test, y_predict)

# defining parameter range 
param_grid = {'C': [0.1, 1, 10, 100, 1000],  
              'gamma': [1, 0.1, 0.01, 0.001, 0.0001], 
              'kernel': ['rbf']}

# Creating Grid of parameters
grid = ParameterGrid(param_grid)
grid_results = []
cm_scores =[]

# Running Grid-CV for hypertuning SVC having low tolerance & early stopping
for params in grid:
    SVC_model1 = SVC(**params, tol=0.01)
    cv_results = cross_val_score(SVC_model1, X_train, y_train, cv=10, scoring='roc_auc')
    cm_scores.append(list(cv_results))
  
# Creating generator for grid_results & finding best result.
grid_results = list(zip(grid, cm_scores))
# print best parameter/result after tuning
best_result = max(grid_results, key=lambda x: x[1])
print(best_result)
best_param = best_result[0]

# Creating SVM Classifier with best parameters
best_svr = SVC(kernel=best_param['kernel'],
               C=best_param['C'],
               gamma=best_param['gamma'],
               coef0=0.1, shrinking=True,
               tol=0.001, cache_size=200, verbose=False, max_iter=-1)

best_svr.fit(X_train, y_train)
grid_predict = best_svr.predict(X_test)
# print classification report 
print(classification_report(y_test, grid_predict))