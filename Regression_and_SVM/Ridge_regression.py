
import numpy as np
from sklearn import datasets
from numpy.linalg import inv
from math import sqrt,pi
import math
import matplotlib.pyplot as plt




def Rigde_func(X,Ytrain,lambd):

    Xt_X = np.dot(np.transpose(X), X)
    LI = lambd*np.eye(Xt_X.shape[0])
    w = np.dot(inv(Xt_X+LI), np.dot(np.transpose(X), Ytrain))
    return w
    w = np.dot(inv(np.dot(np.transpose(X),X)),np.dot(np.transpose(X),Ytrain))
    return w

def RMSE_cal(w,Xtest,ytest):
    
    y_est = np.dot(Xtest,w)
    error = y_est - ytest
    
    mse = np.dot(np.transpose(error),error)/(error.shape[0])
    rmse = sqrt(mse)
    return rmse



diabetes = datasets.load_diabetes()
#print(type(diabetes))
Xtrain =  diabetes.data[:242]
Xtest = diabetes.data[242:]

Ytrain = diabetes.target[:242]
Ytest = diabetes.target[242:]

#print(len(Xtrain))



# intercept
Xtrain_i = np.concatenate((np.ones((Xtrain.shape[0],1)), Xtrain), axis=1)
Xtest_i = np.concatenate((np.ones((Xtest.shape[0],1)), Xtest), axis=1)

k = 6
lambdas = np.linspace(0.000, 0.005, num=k)
i = 0
rmse_train = np.zeros((k,1))
rmse_test = np.zeros((k,1))
for lambd in lambdas:
    w_l = Rigde_func(Xtrain_i,Ytrain,lambd)
    rmse_train[i] = RMSE_cal(w_l,Xtrain_i,Ytrain)
    rmse_test[i] = RMSE_cal(w_l,Xtest_i,Ytest)
    i = i + 1
    #print(w_l)
index = 0
print ("lambda  train             test")
while (index < len(rmse_test)):
    print (np.reshape(lambdas,(lambdas.shape[0],1))[index],rmse_train[index], rmse_test[index])
    index += 1

print(rmse_train)

fig = plt.figure(figsize=[12,6])
plt.subplot(1, 2, 1)
plt.plot(lambdas,rmse_train)
plt.title('RMSE for Train Data')
plt.subplot(1, 2, 2)
plt.plot(lambdas,rmse_test)
plt.title('RMSE for Test Data')
plt.show()