import numpy as np
from scipy.optimize import minimize
from scipy.io import loadmat
from math import sqrt
import time





def initializeWeights(n_in,n_out):
    """
    # initializeWeights return the random weights for Neural Network given the
    # number of node in the input layer and output layer

    # Input:
    # n_in: number of nodes of the input layer
    # n_out: number of nodes of the output layer
       
    # Output: 
    # W: matrix of random initial weights with size (n_out x (n_in + 1))"""
    
    epsilon = sqrt(6) / sqrt(n_in + n_out + 1);
    W = (np.random.rand(n_out, n_in + 1)*2* epsilon) - epsilon;
    return W
    
    
    
def sigmoid(z):
    
    """# Notice that z can be a scalar, a vector or a matrix
    # return the sigmoid of input z"""


    return (1.0/ (1.0 + np.exp(-z)))
    
    

def preprocess():
    """ Input:
     Although this function doesn't have any input, you are required to load
     the MNIST data set from file 'mnist_all.mat'.

     Output:
     train_data: matrix of training set. Each row of train_data contains 
       feature vector of a image
     train_label: vector of label corresponding to each image in the training
       set
     validation_data: matrix of training set. Each row of validation_data 
       contains feature vector of a image
     validation_label: vector of label corresponding to each image in the 
       training set
     test_data: matrix of training set. Each row of test_data contains 
       feature vector of a image
     test_label: vector of label corresponding to each image in the testing
       set

     Some suggestions for preprocessing step:
     - divide the original data set to training, validation and testing set
           with corresponding labels
     - convert original data set from integer to double by using double()
           function
     - normalize the data to [0, 1]
     - feature selection"""
    
   mat = loadmat('mnist_all.mat') #loads the MAT object as a Dictionary
    
    # Pick a reasonable size for validation data
    FEATURE_LENGTH = 784  # This "constant" will actually change after simple feature selection.

    EXPECTED_TRAINING = 60000

    EXPECTED_VALIDATION = 10000

    EXPECTED_TESTING = 10000

    DEV_TRAINING = 5000

    DEV_VALIDATING = 1000

    DEV_TESTING = 1000

    DEBUG = False

    if DEBUG:

        print ("OPERATING IN DEBUG MODE WITH REDUCED DATA SET!")



    # Read the matlab matrix file and store it as a 2D array in nested Python lists.

    
    
    # Your code here




    if DEBUG:

        print ("There are a total of", len(mat), "items in the .mat file:", mat.keys())

    training_sets = []  # Should be a 2D list, 60000 x 784 elements long.

    testing_sets = []  # Should be a 2D list, 60000 x 784 elements long.

    all_60000_training_labels = []  # Should be a 1D list, 60000 (EXPECTED TRAINING) elements long.

    test_label = []  # Should be a 1D list, 10000 (EXPECTED TESTING) elements long.

    for k in mat:

        if k.find("test") != -1:  # It's one of test0 - test9.

            true_label = int(k[-1])

            for m in mat[k]:

                test_label.append(true_label)

                testing_sets.append(m)

        elif k.find("train") != -1:  # It's one of train0 - train9.

            true_label = int(k[-1])

            for m in mat[k]:

                all_60000_training_labels.append(true_label)

                training_sets.append(m)

        else:

            pass # Otherwise it's meta-information about the Matlab file that we don't need.



    # Put the 2D testing and training lists into numpy arrays and normalize them.

    test_data = np.array(testing_sets, dtype='double') / 255

    all_training = np.array(training_sets, dtype='double') / 255

    test_label = np.array(test_label)

    all_60000_training_labels = np.array(all_60000_training_labels)



    # Check that the data is in the form we expect.

    assert test_label.shape == (EXPECTED_TESTING,), test_label.shape

    assert all_training.shape == (EXPECTED_TRAINING, FEATURE_LENGTH)

    assert test_data.shape == (EXPECTED_TESTING, FEATURE_LENGTH)

    assert isinstance(all_training[0][0], np.float64)



    # Implement very simple feature selection,eliminate any features that are identical for every example in training.

    temp = all_training.T

    varied_feature_indices = []

    for row in range(len(temp)):

        if np.unique(temp[row:row+1, ::]).size > 1:

            varied_feature_indices.append(row)

    # Update the constant of our feature length.

    FEATURE_LENGTH = len(temp[varied_feature_indices])

    # Update the training data to exclude useless features.

    temp = temp[varied_feature_indices]

    all_training = temp.T

    # Update the testing data too to exclude useless features.

    # Of course in the real world we could choose which features to eliminate from our test data,

    # but then again in the real world, we would not have our test data!

    temp = test_data.T

    temp = temp[varied_feature_indices]

    test_data = temp.T





    # Now randomly divide the training data into two matrices: one with 50,000 rows, and another with 10,000.

    # The 50,000-row matrix is "training" data, and the 10,000-row matrix is "validating" data.

    # Split their labels at the same time, to make sure that the true labels are at the same index as their data.

    random_training_indices = np.array(random.sample(list(range(EXPECTED_TRAINING)), 50000))

    random_validating_indices = np.array(list(set(list(range(EXPECTED_TRAINING))) - set(random_training_indices)))

    # See http://docs.scipy.org/doc/numpy-1.10.1/user/basics.indexing.html, section "Index Arrays", for an explanation of this syntax on a np array.

    train_data = all_training[random_training_indices]

    validation_data = all_training[random_validating_indices]

    train_label = all_60000_training_labels[random_training_indices]

    validation_label = all_60000_training_labels[random_validating_indices]

    # Double check that data is in expected shape.

    assert train_data.shape == (EXPECTED_TRAINING - EXPECTED_VALIDATION, FEATURE_LENGTH)

    assert validation_data.shape == (EXPECTED_VALIDATION, FEATURE_LENGTH)

    assert train_label.shape == (EXPECTED_TRAINING - EXPECTED_VALIDATION, )

    assert validation_label.shape == (EXPECTED_VALIDATION, )



    if DEBUG:

        # Return a smaller subset of the data for faster development.

        # Note that these are shallow copies http://docs.scipy.org/doc/numpy-1.10.1/reference/generated/numpy.copy.html.

        train_data = train_data[:DEV_TRAINING]

        train_label = train_label[:DEV_TRAINING]

        validation_data = validation_data[:DEV_VALIDATING]

        validation_label = validation_label[:DEV_VALIDATING]

        random_testing_indices = np.array(random.sample(list(range(EXPECTED_TESTING)), DEV_TESTING))

        test_data = test_data[random_testing_indices]

        test_label = test_label[random_testing_indices]

    else:

        pass # Return the really large data matrices.
    
    print ("Returning training data with shape (rows, cols):" + str(train_data.shape))

    print ("Returning labels for training data with shape (rows, cols):" + str(train_label.shape))

    print ("Returning validation data with shape (rows, cols):" + str(validation_data.shape))

    print ("Returning labels for validation data with shape (rows, cols):" + str(validation_label.shape))

    print ("Returning test data with shape (rows, cols):" + str(test_data.shape))

    print ("Returning labels for test data with shape (rows, cols):" + str(test_label.shape))

    print ("Data type of all entries in matrices is:" + str(type(all_training[0][0])))

    return train_data, train_label, validation_data, validation_label, test_data, test_label

    # Creat Normalized Matrix

    



def nnObjFunction(params, *args):
    """% nnObjFunction computes the value of objective function (negative log 
    %   likelihood error function with regularization) given the parameters 
    %   of Neural Networks, thetraining data, their corresponding training 
    %   labels and lambda - regularization hyper-parameter.

    % Input:
    % params: vector of weights of 2 matrices w1 (weights of connections from
    %     input layer to hidden layer) and w2 (weights of connections from
    %     hidden layer to output layer) where all of the weights are contained
    %     in a single vector.
    % n_input: number of node in input layer (not include the bias node)
    % n_hidden: number of node in hidden layer (not include the bias node)
    % n_class: number of node in output layer (number of classes in
    %     classification problem
    % training_data: matrix of training data. Each row of this matrix
    %     represents the feature vector of a particular image
    % training_label: the vector of truth label of training images. Each entry
    %     in the vector represents the truth label of its corresponding image.
    % lambda: regularization hyper-parameter. This value is used for fixing the
    %     overfitting problem.
       
    % Output: 
    % obj_val: a scalar value representing value of error function
    % obj_grad: a SINGLE vector of gradient value of error function
    % NOTE: how to compute obj_grad
    % Use backpropagation algorithm to compute the gradient of error function
    % for each weights in weight matrices.

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % reshape 'params' vector into 2 matrices of weight w1 and w2
    % w1: matrix of weights of connections from input layer to hidden layers.
    %     w1(i, j) represents the weight of connection from unit j in input 
    %     layer to unit i in hidden layer.
    % w2: matrix of weights of connections from hidden layer to output layers.
    %     w2(i, j) represents the weight of connection from unit j in hidden 
    %     layer to unit i in output layer."""
    
    n_input, n_hidden, n_class, training_data, training_label, lambdaval = args
        label_matrix = []

    for label in training_label:

        label_matrix.append([0 if x != label else 1 for x in range(10)])

    label_matrix = np.array(label_matrix)
   
    label_matrix = train_label
    w1 = params[0:n_hidden * (n_input + 1)].reshape( (n_hidden, (n_input + 1)))
    w2 = params[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))
    obj_val = 0
    #Your code here


    bias_1 = np.ones((training_data.shape[0], 1))

    training_data_with_bias = np.concatenate((training_data, bias_1), axis=1)

    hidden_output = sigmoid(np.dot(training_data_with_bias, w1.T))



    bias_2 = np.ones((1, hidden_output.T.shape[1]))

    hidden_output_with_bias = np.concatenate((hidden_output.T, bias_2), axis=0).T

    Feedforward_output = sigmoid(np.dot(hidden_output_with_bias, w2.T))



    A = np.ones((np.shape(label_matrix)[0], 10))

    Item_1 = np.multiply(label_matrix, np.log(Feedforward_output))

    Item_2 = np.multiply(A - label_matrix, np.log(A - Feedforward_output))

    M = Item_1 + Item_2

    J_w1_w2 = (-1 * np.sum(np.sum(M, axis=0)))/training_data.shape[0]



    w1_squared = np.dot(w1.flatten(), w1.flatten().T)

    w2_squared = np.dot(w2.flatten(), w2.flatten().T)

    regularization_term = np.dot(lambdaval, (w1_squared + w2_squared))

    obj_val = J_w1_w2.flatten() + regularization_term / np.dot(2, training_data.shape[0])


    
    delta_l = np.array(label_matrix - Feedforward_output) # correspondes to eqn(9)

    dev_lj = -1*np.dot(delta_l.T, hidden_output_with_bias) # correspondes to eqn(8)

    grad_w2 = (dev_lj + lambdaval *w2)/ training_data.shape[0] #correspondes to eqn(16)

    w2_noBias = w2[:,0:-1]

    delta_j = np.array(hidden_output)*np.array(1-hidden_output) # correspondes to -(1-Zj)Zj in eqn(12)

    dev_ji = -1*np.dot((np.array(delta_j)*np.array(np.dot(delta_l,w2_noBias))).T,training_data_with_bias) # correspondes to eqn(12)

    grad_w1 = (dev_ji+lambdaval*w1)/training_data.shape[0] #correnspondes to eqn(17)



    # Reshape the gradient matrices to a 1D array.

    grad_w1_reshape = np.ndarray.flatten(grad_w1.reshape((grad_w1.shape[0]*grad_w1.shape[1],1)))

    grad_w2_reshape = grad_w2.flatten()

    obj_grad_temp = np.concatenate((grad_w1_reshape.flatten(), grad_w2_reshape.flatten()),0)

    obj_grad = np.ndarray.flatten(obj_grad_temp)

    return (obj_val,obj_grad)




def nnPredict(w1,w2,data):
    
    """% nnPredict predicts the label of data given the parameter w1, w2 of Neural
    % Network.

    % Input:
    % w1: matrix of weights of connections from input layer to hidden layers.
    %     w1(i, j) represents the weight of connection from unit i in input 
    %     layer to unit j in hidden layer.
    % w2: matrix of weights of connections from hidden layer to output layers.
    %     w2(i, j) represents the weight of connection from unit i in input 
    %     layer to unit j in hidden layer.
    % data: matrix of data. Each row of this matrix represents the feature 
    %       vector of a particular image
       
    % Output: 
    % label: a column vector of predicted labels""" 

    input_bias = np.ones((data.shape[0],1))  # create a bias

    data_bias = np.concatenate((data, input_bias), axis=1)  # add bias to training data

    hiden_out = sigmoid(np.dot(data_bias, w1.T))  # 3.32 equtions 1 and 2

    hiden_bias = np.ones((1,hiden_out.T.shape[1]))  # create a bias

    hiden_out_bias = np.concatenate((hiden_out.T, hiden_bias), axis=0).T  # add bias to hidden_out data

    net_out = sigmoid(np.dot(hiden_out_bias,w2.T))  # 3.32 eqution 3 and 4, feed forward is complete.

    # Make a 1D vector of the predictions.

    return net_out.argmax(axis=1)

"""**************Neural Network Script Starts here********************************"""
start=time.time()
train_data, train_label, validation_data,validation_label, test_data, test_label = preprocess();

#transfer the y to a 10-dimension vector
def vectorized_result(y):
    e=np.zeros(10)
    e[y]=1.0
    return e
y_tdvec=np.array([vectorized_result(y) for y in train_label])

print(train_label)
print(y_tdvec)

#  Train Neural Network

# set the number of nodes in input unit (not including bias unit)
n_input = train_data.shape[1];

# set the number of nodes in hidden unit (not including bias unit)
n_hidden = 10;
				   
# set the number of nodes in output unit
n_class = 10;				   

# initialize the weights into some random matrices
initial_w1 = initializeWeights(n_input, n_hidden);
initial_w2 = initializeWeights(n_hidden, n_class);


# unroll 2 weight matrices into single column vector
initialWeights = np.concatenate((initial_w1.flatten(), initial_w2.flatten()),0)

# set the regularization hyper-parameter
lambdaval = 0.2;


args = (n_input, n_hidden, n_class, train_data, y_tdvec, lambdaval)

#Train Neural Network using fmin_cg or minimize from scipy,optimize module. Check documentation for a working example

opts = {'maxiter': 50}    # Preferred value.



nn_params = minimize(nnObjFunction, initialWeights, jac=True, args=args,method='CG', options=opts)

#In Case you want to use fmin_cg, you may have to split the nnObjectFunction to two functions nnObjFunctionVal
#and nnObjGradient. Check documentation for this function before you proceed.
#nn_params, cost = fmin_cg(nnObjFunctionVal, initialWeights, nnObjGradient,args = args, maxiter = 50)



#Reshape nnParams from 1D vector into w1 and w2 matrices
w1 = nn_params.x[0:n_hidden * (n_input + 1)].reshape( (n_hidden, (n_input + 1)))
w2 = nn_params.x[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))


#Test the computed parameters

predicted_label = nnPredict(w1,w2,train_data)

#find the accuracy on Training Dataset

print('\n Training set Accuracy:' + str(100*np.mean((predicted_label == train_label).astype(float))) + '%')

predicted_label = nnPredict(w1,w2,validation_data)

#find the accuracy on Validation Dataset

print('\n Validation set Accuracy:' + str(100*np.mean((predicted_label == validation_label).astype(float))) + '%')


predicted_label = nnPredict(w1,w2,test_data)

#find the accuracy on Validation Dataset

print('\n Test set Accuracy:' +  str(100*np.mean((predicted_label == test_label).astype(float))) + '%')

end=time.time()
print(end-start)
