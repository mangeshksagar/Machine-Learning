#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 28 22:01:05 2019

@author: mangesh.kshirsagar
"""

import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer 
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV
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

# Running Grid-CV
grid = GridSearchCV(SVC(), param_grid, refit = True, verbose = 3) 
  
# fitting the model for grid search 
grid.fit(X_train, y_train)

# print best parameter after tuning 
print(grid.best_params_)
best_params = grid.best_params_
best_svr = SVC(kernel='rbf',
               C=best_params["C"],
               gamma=best_params["gamma"],
               coef0=0.1, shrinking=True,
               tol=0.001, cache_size=200, verbose=False, max_iter=-1)

# print how our model looks after hyper-parameter tuning 
print(grid.best_estimator_) 
grid_predictions = grid.predict(X_test)

# print classification report 
print(classification_report(y_test, grid_predictions))
