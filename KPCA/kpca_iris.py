import numpy as np
import pandas as pd
from scipy import linalg as LA
dataset = pd.read_csv("/nfs/cms/mtech18/mangesh.kshirsagar/Desktop/ML-2/KPCA/iris.csv")
dataset = dataset.iloc[0:100,:]
train_data = dataset.drop(["Id","Species"],axis=1).values
target = dataset.Species.map({"Iris-setosa":0,"Iris-versicolor":1,"Iris-virginica":2})
def polyn_krnl(a,b):
    return (1 + np.dot(a,b))**2

def Gaussian_kernal(x,xt):
    return (np.exp(-(np.sum(np.abs(x-xt)**2)/2)))

K = polyn_krnl(train_data,np.transpose(train_data))

a = (1.0/100) * np.ones([100,100])


K_hat = np.mat(K)- np.mat(a)*np.mat(K) - np.mat(K)*np.mat(a) + np.mat(a)*np.mat(K)*np.mat(a)
print K_hat[:,0].mean()
eigen_val,eigen_vec = LA.eig(K_hat)
eigen_val = np.real(eigen_val)
eigen_vec =  np.real(eigen_vec)

conv_data = np.dot(eigen_vec,train_data)
aa  = pd.DataFrame(conv_data)
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(conv_data,target.values,test_size=0.2,random_state=42)

from sklearn.ensemble import RandomForestClassifier
rnd_clf = RandomForestClassifier()
rnd_clf.fit(X_train,y_train)
y_pred = rnd_clf.predict(X_test)
from sklearn.metrics import accuracy_score,make_scorer
accuracy = accuracy_score(y_test,y_pred)
print accuracy*100
from sklearn.model_selection import cross_val_score
score = make_scorer(accuracy_score)
clf = cross_val_score(estimator=rnd_clf,X=conv_data,y=target.values,cv=5,scoring=score)
#cov_mat.append((1 + np.dot(X_train1[i,:],X_train1[j,:].T))**2 )