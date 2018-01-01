import numpy as np
import h5py
import matplotlib.pyplot as plt
from testCases_v2 import *
from dnn_utils_v2 import sigmoid, sigmoid_backward, relu, relu_backward

def initializeParameter(sizeInput , sizeHidden , sizeOutPut):
    np.random.seed(1)
    W1 = np.random.rand(sizeHidden, sizeInput)*0.01
    b1 = np.zeros(shape=(sizeHidden, 1))
    W2 = np.random.rand( sizeOutPut, sizeHidden)*0.01
    b2 = np.zeros(shape=(sizeOutPut, 1))
    parameter = {"W1" : W1 , "b1":b1 , "W2":W2 , "b2":b2}
    return parameter

def initializeParameterDeep(layerDim):
    length = len(layerDim)                  #layerDim is an array with the size of each layer (number of neurons) in our architecture
    parameters = {}
    for k in range(1, length):
        parameters["W"+str(k)] = np.random.rand(layerDim[k], layerDim[k-1])*0.01
        parameters["b"+str(k)] = np.zeros(shape=(layerDim[k], 1))
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
    L = parameters
    for k in range(1 , L):
        Aprev = A
        A , cache = linearActivationForward(Aprev, parameters["W"+str(k)], parameters["b"+str(k)], "relu")
        caches.append(cache)
    AL, cache =linearActivationForward(A, parameters["WL"], parameters["bL"], "sigmoid")
    caches.append(cache)
    # caches contain (data=A1,W1,b1,Z1) , (relu(Z1)= A2,W2,b2) .... (sigmoid(Z[L-2]=A[L-1],W[L-1],b[L-1])
    return AL, caches

def CostFunction(AL,Y):
    cost = - np.sum(np.multiply(Y,np.log(AL)) + np.multiply(1-Y,np.log(1-AL))) / Y.shape[1]
    return np.squeeze(cost)

def linearBackward(dZ , cache):
    A_prev , W , b = cache
    m = A_prev.shape[1]
    dW = (1 / m) * np.dot(dZ ,A_prev.T )
    db = (1 / m) * np.squeeze(np.sum(dZ, axis=1, keepdims=True))
    dA_prev = np.dot(W.T , dZ)
    return dA_prev,dW,db

def linearActivationBackward(dA,cache , activation):
    if   activation == "relu":
        dZ = relu_backward(dA,cache)
    elif activation == "sigmoid":
        dZ = sigmoid_backward(dA,cache)
    return linearBackward(dZ, cache)

def backPropogation(AL, Y , caches):
    # AL - probability vector, output of the forward propagation(L_model_forward())
    # Y  - true "label" vector(containing 0 if non - cat, 1 if cat)
    # caches  - list of caches containing: every  cache  of linear_activation_forward()
    # with "relu"(it's caches[l], for l in range(L-1) i.e l = 0...L-2)
    # the cache of linear_activation_forward() with "sigmoid" (it's caches[L-1])
    grads = {}
    L = len(caches)
    dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
    grads["dA"+str(L)], grads["dW"+str(L)], grads["db"+str(L)] = linearActivationBackward(dAL, caches[-1], activation="sigmoid")
    for k in reversed(range(L - 1)):
        dA_prev_temp, dW_temp, db_temp = linearActivationBackward(grads["dA"+str(k+2)], caches[k], activation="relu")
        grads["dA"+str(k+1)] = dA_prev_temp
        grads["dW"+str(k+1)] = dW_temp
        grads["db"+str(k+1)] = db_temp
    return grads


def update_parameters(parameters, grads, learning_rate):
    L = len(parameters) // 2  # number of layers in the neural network - we must remember that parameter contain w and b.
    for k in range(L):
        parameters["W{}".format(k+1)] = parameters["W{}".format(k+1)] - grads["dW{}".format(k+1)] * learning_rate
        parameters["b{}".format(k + 1)] = parameters["b{}".format(k + 1)] - grads["db{}".format(k + 1)] * learning_rate
    return parameters



