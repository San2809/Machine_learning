import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.optimize import minimize
from sklearn.svm import SVC
from sklearn import preprocessing

Xtrain = loadmat('ijcnn_train.mat')
Xtest = loadmat('ijcnn_test.mat')

#scale = MinMaxScaler(copy=True, feature_range=(0,1))
#print(Xtrain)


train_data = Xtrain['ijcnn_data']
train_label = Xtrain['ijcnn_label']

test_data = Xtest['test_data']
test_label = Xtest['test_label']

#print(type(train_data))

norm_train = preprocessing.normalize(train_data)

print('SVM\nLinear Kernel')

train_label = np.squeeze(train_label)

clf = SVC(kernel = 'linear')

clf.fit(train_data, train_label)

print("Training Accuracy:   "+str(100*clf.score(train_data, train_label))+"%")

print("Test Accuracy:       "+str(100*clf.score(test_data, test_label))+"%")




print ("\n\n RBF Gamma =1\n");

clf = SVC(kernel='rbf',gamma = 1)

clf.fit(train_data, train_label)

print("Training Accuracy:   "+str(100*clf.score(train_data, train_label))+"%")

print("Test Accuracy:       "+str(100*clf.score(test_data, test_label))+"%")




print ("\n\nRBF default \n");

clf = SVC(kernel = 'rbf')

clf.fit(train_data, train_label)

print("Training Accuracy:   "+str(100*clf.score(train_data, train_label))+"%")

print("Test Accuracy:       "+str(100*clf.score(test_data, test_label))+"%")






cvalues  = [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

TrainingAccuracy = []

TestAccuracy = []



print ("---------------------------------------------------")

for i in cvalues:

    train_label = np.squeeze(train_label)

    clf = SVC(C=i,kernel='rbf')

    print("Doing for C-Value: ",i)

    clf.fit(train_data, train_label)

    print("CLF Fitting Done!")

    print("Training Accuracy:   ",100*clf.score(train_data, train_label),"%")

    print("Testing Accuracy:    ",100*clf.score(test_data, test_label),"%")


    TrainingAccuracy.append(100*clf.score(train_data, train_label))

    TestAccuracy.append(100*clf.score(test_data, test_label))


    print ("---------------------------------------------------")

    
accuracyMatrix = np.column_stack((TrainingAccuracy, TestAccuracy))



fig = plt.figure(figsize=[12,6])

plt.subplot(1, 2, 1)

plt.plot(cvalues,accuracyMatrix)

plt.title('Accuracy with varying values of C')

plt.legend(('Testing data','Training data'), loc = 'best')

plt.xlabel('C values')

plt.ylabel('Accuracy in %')

plt.show()


