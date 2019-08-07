#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 21:57:43 2019

@author: mangesh.kshirsagar
"""
#%%imorting required packages:-
import numpy as np
from sklearn import preprocessing ,cross_validation,neighbors
import pandas as pd
df=pd.read_csv("/nfs/cms/mtech18/mangesh.kshirsagar/Desktop/ML-2/breast-cancer-wisconsin.data")
df.replace('?',-99999,inplace=True)
df.drop(['id'],1,inplace=True)
#%%
x=np.array(df.drop(['Class'],1)) 
y=np.array(df['Class'])
#%%
x_train, x_test, y_train, y_test = cross_validation.train_test_split(x,y,test_size=0.2)
#%%
clf=neighbors.KNeighborsClassifier()
clf.fit(x_train,y_train)
#%%
accuracy=clf.score(x_test,y_test)
print("Model accuracy is ", accuracy*100)
#%%
example_measures=np.array([8,10,10,8,7,10,9,7,1])
example_measures=example_measures.reshape(1,-1)
#%%
prediction=clf.predict(example_measures)
print("The Given exmaple is the type of ",prediction)
#%%