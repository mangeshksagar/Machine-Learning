#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  2 06:08:04 2019

@author: mangesh.kshirsagar
"""
#%%
from __future__ import division, print_function
import numpy as np
from sklearn import datasets,svm
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt
#%%
iris=datasets.load_iris()
X=iris.data[:,:2]
y=iris.target
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=42)
#%%
def evaluate_on_test_data(model=None):
    predictions=model.predict(X_test)
    misclassification=0
    for i in range(len(y_test)):
        if predictions[i]==y_test[i]:
            misclassification += 1
    accuracy = 100*misclassification/len(y_test)
    return accuracy
#%%
kernels=('linear','poly','rbf')
accuracies=[]
for index, kernel in enumerate(kernels):
    model=svm.SVC(kernel=kernel)
    model.fit(X_train,y_train)
    acc=evaluate_on_test_data(model)
    accuracies.append(acc)
    print("{} % accuracy obtained with kernel ={}".format(acc,kernel))
#%% Train SVMs with different kernels
    svc=svm.SVC(kernel='linear').fit(X_train,y_train)
    rbf_svc=svm.SVC(kernel='rbf',gamma=0.7).fit(X_train,y_train)
    poly_svc=svm.SVC(kernel='poly',degree=3).fit(X_train,y_train)
#create a mesh to plot in 
h=0.2 #step size in mesh    
x_min,x_max=X[:,0].min() -1, X[:,0].max() + 1
y_min,y_max=X[:,1].min() -1, X[:,1].max() + 1
xx,yy=np.meshgrid(np.arange(x_min,x_max,h),np.arange(y_min,y_max))
#define title for the plots
titles= ['SVC with kernel','SVC with RBF kernel','SVC with polynomial (degree3) kernel']
#%%
for i, clf in enumerate((svc,rbf_svc,poly_svc)):
    plt.figure(i)
    
    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, m_max]x[y_min, y_max].
    Z = clf.predict(np.c_[xx.ravel(),yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx,yy,Z,cmap=plt.cm.Paired,aplha=0.8)
    plt.scatter(X[:,0],X[:,1],c=y,cmap=plt.cm.ocean)
    plt.ylabel("Sepal Width")
    plt.xlabel("Sepal Length")
    plt.xlim(xx.min(),xx.max())
    plt.ylim(yy.min(),yy.max())
    plt.xticks(())
    plt.yticks(())
    plt.title(titles[i])
plt.show()
#%%
#Checking the support vectors of the polynomial kernel (for example)
print("The support vectors are:\n", poly_svc.support_vectors_)
#%%