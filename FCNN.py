import numpy as np
import pickle
import json
import os
import sys

class FCFFNN(object):

    def weight_initialization(self , n_in , n_out , activation="sigmoid"):
        if activation.lower() == "relu":
            return (np.random.randn(n_in , n_out)*np.sqrt(2/n_in))
        else:
            return (np.randn(n_in , n_out)*np.sqrt(1/n_in))

    def __init__(self , layer_size , activation="sigmoid"):

        self._activation_ = activation
        self.layers = layer_size
        self.num_layers = len(layer_size)
        self.input_layer , self.output_layer , self.hidden_layers = self.layers[0] , self.layers[-1] , self.layers[1:-1]
        self.weights , self.biases = [] , []

        for i in range(self.num_layers-1):
            self.weights.append( self.weight_initialization(self.layers[i+1] , self.layers[i] , activation) )
            self.biases.append( np.zeros((self.layers[i+1] , 1)) )

    def Loss_CE(self , AL , y , Catagorial=True):
        AL = np.clip(AL,1e-12,1-1e-12)
        if Catagorial:
            return -np.sum(y*np.log(AL))/AL.shape[1]
        else:
            return -np.sum(y*np.log(AL) + (1-y)np.log(1-AL))/AL.shape[1]

    def Loss_sq(self , AL , y):
        return np.mean((AL-y)**2)

    def sigmoid(Z):
        return 1/(1+np.exp(-(np.clip(Z,-500,500))))
    def tanh(Z):
        return np.tanh(Z)
    def ReLU(Z):
        return np.maximum(0 , Z)

    def dSig(Z):
        return FCFFNN.sigmoid(Z)*(1-FCFFNN.sigmoid(Z))
    def dtanh(Z):
        return 1-(np.tanh(Z)**2)
    def dReLU(Z):
        return (Z>0).astype(float)
    def dsame(Z):
        return np.ones_like(Z)

    def softmax(Z):
        shift_Z = Z - np.max(Z , axis=0 , keepdims=True)
        return np.exp(shift_Z)/np.sum(np.exp(shift_Z) , axis=0 , keepdims=True)
    def same(Z):
        return Z

    def lr_constant(t , k=0.01):
        return k