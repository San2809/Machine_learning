import numpy as np
from sklearn import datasets
from numpy.linalg import inv
from math import sqrt,pi
import math
import matplotlib.pyplot as plt
from scipy.optimize import minimize

def Ridge_func(X,Ytrain,lambd):
    Xt_X = np.dot(np.transpose(X), X)
    LI = lambd*np.eye(Xt_X.shape[0])
    w = np.dot(inv(Xt_X+LI), np.dot(np.transpose(X), Ytrain))
    return w
    w = np.dot(inv(np.dot(np.transpose(X),X)),np.dot(np.transpose(X),Ytrain))
    return w



def RMSE_cal(w,Xtest,Ytest):
    
    xw=np.dot(Xtest,w)   
    yxw= Ytest-xw    
    mse=np.dot(np.transpose(yxw),yxw)/Xtest.shape[0]
    rmse = np.sqrt(mse)
    
    return rmse




def objFunction(w, Xtrain_i, Ytrain, lambd):

    error =0.5*(np.dot(np.transpose(np.subtract(Ytrain,np.dot(Xtrain_i,w))),np.subtract(Ytrain,np.dot(Xtrain_i,w))))+0.5*lambd*(np.dot(np.transpose(w),w))  

    


    error_grad = np.dot(np.dot(np.transpose(w), np.transpose(Xtrain_i)), Xtrain_i) - np.dot(np.transpose(Ytrain),Xtrain_i) + np.transpose(lambd * w)
  

    error_grad = np.squeeze(np.array(error_grad))

    

    return error, error_grad



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
    w_l = Ridge_func(Xtrain_i,Ytrain,lambd)
    rmse_train[i] = RMSE_cal(w_l,Xtrain_i,Ytrain)
    rmse_test[i] = RMSE_cal(w_l,Xtest_i,Ytest)
    i = i + 1
    





k = 6
lambdas = np.linspace(0.000, 0.005, num=k)
i = 0
rmse2_train = np.zeros((k,1))
rmse2_test = np.zeros((k,1))
opts = {'maxiter' : 110}    # Preferred value.                                                
w_init = np.zeros((Xtrain_i.shape[1],1))
#print(len(w_init))
#print(objFunction(w_init,Xtrain_i,Ytrain,0.002))
for lambd in lambdas:
    args = (Xtrain_i, Ytrain, lambd)
    w_l = minimize(objFunction, w_init, jac=True, args=args,method='CG', options=opts)
    w_l = np.transpose(np.array(w_l.x))
    w_l = np.reshape(w_l,[len(w_l),1])
    w_l = w_l.flatten()
 
    #print(w_l)
    rmse2_train[i] = RMSE_cal(w_l,Xtrain_i,Ytrain)
    rmse2_test[i] = RMSE_cal(w_l,Xtest_i,Ytest)
    i = i + 1

 #find minimum lambda
index = 0
print ("lambda        train mse      test mse")
while (index < len(w_l)):
    print (lambdas[index],"   ",rmse2_train[index], "   ", rmse2_test[index])
    index += 1
    if (index > 5):
        break
#print(rmse_train)
#print(mses4_train) 
 
fig = plt.figure(figsize=[12,6])
plt.subplot(1, 2, 1)
plt.ylim(53.75,54.05)
plt.plot(lambdas,rmse2_train)
plt.plot(lambdas,rmse_train)
plt.title('RMSE for Train Data')
plt.legend(['Using Gradient descent','Without GD'])

plt.subplot(1, 2, 2)
plt.plot(lambdas,rmse2_test)
plt.plot(lambdas,rmse_test)
plt.title('RMSE for Test Data')
plt.legend(['Using Gradient Descent','without GD'])
plt.show()


