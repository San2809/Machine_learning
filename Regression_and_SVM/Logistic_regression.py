import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.optimize import minimize
from sklearn.svm import SVC
from sklearn import preprocessing

Xtrain = loadmat('ijcnn_train.mat')
Xtest = loadmat('ijcnn_test.mat')

#scale = MinMaxScaler(copy=True, feature_range=(0,1))


train_data = Xtrain['ijcnn_data'].toarray()
train_label = Xtrain['ijcnn_label']

test_data = Xtest['test_data'].toarray()
test_label = Xtest['test_label']







def ObjFunction(initialWeights, *args):

    train_data, labeli = args

    #print ("num samples  and feature -- ",train_data.shape[0],train_data.shape[1])

    n_data = train_data.shape[0]

    n_features = train_data.shape[1]

    # Adding bias term

    BiasTerm = np.ones((train_data.shape[0],1))

    Bias = np.column_stack((BiasTerm, train_data))

    # print (" shape ",Bias.shape[0],Bias.shape[1])

    W = initialWeights.reshape(Bias.shape[1],1)

    Theta = np.zeros((Bias.shape[0],1))

    Theta = sigmoid(np.dot(Bias,W))

    # print (" thetaa shape ",Theta.shape[0],Theta.shape[1])

    LogTheta = np.log(Theta)

    y=np.dot(labeli.transpose(),LogTheta)



    # implement the formula for error

    part1 =  np.dot(labeli.transpose(), LogTheta)

    part2 = np.dot(np.subtract(1.0,labeli).transpose(), np.log(np.subtract(1.0,Theta)))

    error = np.sum(part1 + part2)

    error = (-error)/Bias.shape[0]



    # Implement the formula for error_grad

    error_grad = np.zeros((Bias.shape[0], 1))

    part3 = np.zeros((Bias.shape[0],1))

    part3 = np.subtract(Theta,labeli)

    error_grad = np.dot(Bias.transpose(), part3)

    error_grad=error_grad/Bias.shape[0]



    return error, error_grad.reshape((n_feature+1,))








def prediction_func(W, data):

    Num_of_Samples = data.shape[0]

    label = np.zeros((Num_of_Samples, 1))

    BiasTerm = np.ones((Num_of_Samples,1))

    Bias = np.column_stack((BiasTerm, data))

    # print ("shape is ",Bias.shape[0],Bias.shape[1])

    All_Labels = np.zeros((Bias.shape[0],))

    All_Labels = sigmoid(np.dot(Bias,W))

    # use argmax and find the lable with the max value

    label = np.argmax(All_Labels, axis =1)

    # print (" shape of label is ",label.shape[0],label.shape[1])

    # reshpae and return the label

    label = label.reshape(Num_of_Samples,1)

    return label


def sigmoid(z):

    return 1.0 / (1.0 + np.exp(-z))


# number of classes

n_class = 10



# number of training samples

n_train = train_data.shape[0]



# number of features

n_feature = train_data.shape[1]



Y = np.zeros((n_train, n_class))

for i in range(n_class):

    Y[:, i] = (train_label == i).astype(int).ravel()



# Logistic Regression with Gradient Descent

W = np.zeros((n_feature + 1, n_class))

initialWeights = np.zeros((n_feature + 1, 1))

opts = {'maxiter': 100}

for i in range(n_class):

    labeli = Y[:, i].reshape(n_train, 1)

    args = (train_data, labeli)

    linear_params = minimize(ObjFunction, initialWeights, jac=True, args=args, method='CG', options=opts)

    W[:, i] = linear_params.x.reshape((n_feature + 1,))



# Find the accuracy on Training Dataset

predicted_label = prediction_func(W, train_data)

print('\n Logistic Regression')

print('\n Training set Accuracy:' + str(1000 * np.mean((predicted_label == train_label).astype(float))) + '%')






# Find the accuracy on Testing Dataset

predicted_label = prediction_func(W, test_data)

print('\n Testing set Accuracy:' + str(1000 * np.mean((predicted_label == test_label).astype(float))) + '%')