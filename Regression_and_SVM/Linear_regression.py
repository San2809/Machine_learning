
import numpy as np
from sklearn import datasets
from numpy.linalg import inv
from math import sqrt,pi
import math
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from sklearn.svm import SVC
from sklearn import preprocessing
from scipy.io import loadmat






def linear_reg(Xtrain,Ytrain):
    w = np.dot(inv(np.dot(np.transpose(Xtrain),Xtrain)),np.dot(np.transpose(Xtrain),Ytrain))
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
print('Problem 1 ')

print ('Training Data ')
w = linear_reg(Xtrain,Ytrain)
rmse = RMSE_cal(w,Xtrain,Ytrain)

w_i = linear_reg(Xtrain_i,Ytrain)
rmse_i = RMSE_cal(w_i,Xtrain_i,Ytrain)

print('RMSE without intercept '+str(rmse))
print('RMSE with intercept '+str(rmse_i))

print ('Test Data ')
w2 = linear_reg(Xtrain,Ytrain)
rmse2 = RMSE_cal(w2,Xtest,Ytest)

w_i2 = linear_reg(Xtrain_i,Ytrain)
rmse_i2 = RMSE_cal(w_i2,Xtest_i,Ytest)

print('RMSE without intercept '+str(rmse2))
print('RMSE with intercept '+str(rmse_i2))

