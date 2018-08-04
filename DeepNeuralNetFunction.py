import numpy as np
import h5py
import matplotlib.pyplot as plt
from sklearn.svm.libsvm import predict

from data import load_dataset
from testCases_v2 import *
from dnn_utils_v2 import sigmoid, sigmoid_backward, relu, relu_backward



def initializeParameterDeep(layerDim):
    np.random.seed(3) ##******************************************
    length = len(layerDim)                  #layerDim is an array with the size of each layer (number of neurons) in our architecture
    parameters = {}
    for k in range(1, length):
        parameters["W"+str(k)] = np.random.rand(layerDim[k], layerDim[k-1])*0.01
        parameters["b"+str(k)] = np.zeros((layerDim[k],1))
    return parameters                       #Array with all the matrix Wi and vector bi of all the architecture

def linearForward(prevLayerActivation , weightMatrix , biasVector): #linear part of activation function - Z=WA+b
    Z = np.dot( weightMatrix , prevLayerActivation ) + biasVector
    cache = (prevLayerActivation , weightMatrix , biasVector)
    return Z , cache

def linearActivationForward(A_prev , W ,b , activation):
    if activation == "sigmoid":
        Z , linearCache = linearForward(A_prev , W , b)
        outputActivationFunc , activationCache = sigmoid(Z)
    elif activation == "relu":
        Z, linearCache = linearForward(A_prev, W, b)
        outputActivationFunc, activationCache = relu(Z)
    cache = (linearCache , activationCache)
    # outputActivationFunc = sigmoid/relu(Z) = A
    # cache = (linearCache = (A_prev,W,b) , activationCache = Z)
    return outputActivationFunc , cache

def L_modelForward(data , parameters):
    caches = []
    A = data
    L = len(parameters)//2
    for k in range(1 , L):
        Aprev = A
        A , cache = linearActivationForward(Aprev, parameters["W"+str(k)], parameters["b"+str(k)], "relu")
        caches.append(cache)
    AL, cache = linearActivationForward(A, parameters["W"+str(L)], parameters["b"+str(L)], "sigmoid")
    caches.append(cache)
    # caches contain (data=A1,W1,b1,Z1) , (relu(Z1)= A2,W2,b2) .... (sigmoid(Z[L-2]=A[L-1],W[L-1],b[L-1])
    return AL, caches

def CostFunction(AL,Y):
    cost = - np.sum(np.multiply(Y,np.log(AL)) + np.multiply(1-Y,np.log(1-AL))) / Y.shape[1]
    cost = np.squeeze(cost)
    return cost

def linearBackward(dZ , cache):
    A_prev , W , b = cache
    m = A_prev.shape[1]
    dW = (1 / m) * np.dot(dZ ,cache[0].T )
    db = (1 / m) * np.squeeze(np.sum(dZ))
    dA_prev = np.dot(cache[1].T , dZ)

    assert (dA_prev.shape == A_prev.shape)
    assert (dW.shape == W.shape)


    return dA_prev,dW,db

def linearActivationBackward(dA,cache , activation):
    linear_cache, activation_cache = cache
    if   activation == "relu":
        dZ = relu_backward(dA,activation_cache)
    elif activation == "sigmoid":
        dZ = sigmoid_backward(dA,activation_cache)

    dA_prev, dW, db = linearBackward(dZ, linear_cache)
    return dA_prev, dW, db

def backPropogation(AL, Y , caches):
    # AL - probability vector, output of the forward propagation(L_model_forward())
    # Y  - true "label" vector(containing 0 if non - cat, 1 if cat)
    # caches  - list of caches containing: every  cache  of linear_activation_forward()
    # with "relu"(it's caches[l], for l in range(L-1) i.e l = 0...L-2)
    # the cache of linear_activation_forward() with "sigmoid" (it's caches[L-1])

    grads = {}
    L = len(caches)
    Y = Y.reshape(AL.shape)

    dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))

    current_cache = caches[-1]
    grads["dA" + str(L)], grads["dW" + str(L)], grads["db" + str(L)] = linearBackward(sigmoid_backward(dAL,current_cache[1]),current_cache[0])
    for k in reversed(range(L - 1)):
        current_cache = caches[k]
        dA_prev_temp, dW_temp, db_temp = linearBackward(sigmoid_backward(dAL, current_cache[1]), current_cache[0])
        grads["dA"+str(k+1)] = dA_prev_temp
        grads["dW"+str(k+1)] = dW_temp
        grads["db"+str(k+1)] = db_temp
    return grads


def update_parameters(parameters, grads, learning_rate):
    L = len(parameters) // 2  # number of layers in the neural network - we must remember that parameter contain w and b.
    for k in range(L):
        parameters["W" + str(k + 1)] = parameters["W" + str(k + 1)] - learning_rate * grads["dW" + str(k + 1)]
        parameters["b"+str(k + 1)] = parameters["b"+str(k + 1)] - grads["db"+str(k + 1)] * learning_rate
    return parameters


def L_layer_model(X, Y, layers_dims, learning_rate=0.0075, num_iterations=3000, print_cost=False):
    np.random.seed(1)
    costs = []  # keep track of cost
    parameters = initializeParameterDeep(layers_dims)

    # Loop (gradient descent)
    for i in range(0, num_iterations):
        # Forward propagation: [LINEAR -> RELU]*(L-1) -> LINEAR -> SIGMOID.
        AL, caches = L_modelForward(X, parameters)
        # Compute cost.
        cost = CostFunction(AL, Y)
        # Backward propagation.
        grads = backPropogation(AL, Y, caches)
        # Update parameters.
        parameters = update_parameters(parameters, grads, learning_rate)
        # Print the cost every 100 training example
        if print_cost and i % 100 == 0:
            print("Cost after iteration %i: %f" % (i, cost))
        if print_cost and i % 100 == 0:
            costs.append(cost)

    # plot the cost
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per tens)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()

    return parameters

def main():
    train_x_orig, train_y, test_x_orig, test_y, classes = load_dataset()
    train_x_flatten = train_x_orig.reshape(train_x_orig.shape[0],-1).T  # The "-1" makes reshape flatten the remaining dimensions
    test_x_flatten = test_x_orig.reshape(test_x_orig.shape[0], -1).T
    train_x = train_x_flatten / 255.
    test_x = test_x_flatten / 255.
    layers_dims = [12288, 20, 7, 5, 1]
    parameters = L_layer_model(train_x, train_y, layers_dims, num_iterations=1000, print_cost=True)
    predictions_train = predict(train_x, train_y, parameters)
    pred_test = predict(test_x, test_y, parameters)
    print(pred_test)
    print(predictions_train)

if __name__ == "__main__":
    main()

