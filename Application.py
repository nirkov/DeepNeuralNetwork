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
        parameters["W"+length] = np.random.rand(layerDim[k], layerDim[k-1])*0.01
        parameters["b"+length] = np.zeros(shape=(layerDim[k], 1))
    return parameters

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
    return outputActivationFunc , cache

def L_modelForward(data , parameters):
    caches = []
    A = data
    L = parameters
    for k in range(1 , L):
        Aprev = A
        A , cache = linearActivationForward(Aprev,parameters["W"+str(k)],parameters["b"+str(k)] , "relu")
        caches.append(cache)
    AL , cache =  linearActivationForward(A, parameters["WL"], parameters["bL"], "sigmoid")
    caches.append(cache)
    return AL,caches



