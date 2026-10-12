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

    def forward_pass(self , X,Y=None , activation="sigmoid" , lla="softmax" , Sloss="cross entropy"):
        if activation.lower() == "sigmoid":
            self.activation_f = FCFFNN.sigmoid
        elif activation.lower() == "tanh":
            self.activation_f = FCFFNN.tanh
        else:
            self.activation_f = FCFFNN.ReLU

        if lla.lower() == "softmax":
            self.activation_llf = FCFFNN.softmax
        elif lla.lower() == "linear":
            self.activation_llf = FCFFNN.same
        else:
            self.activation_llf = self.activation_f

        if Sloss.lower() == "cross entropy":
            self._loss = self.Loss_CE
        elif Sloss.lower() in ["squared","mse","mean squared error"]:
            self._loss = self.Loss_sq

        self.activations , self.Zs = [X] , []
        for i in range(len(self.weights)):
            A = self.activations[-1]
            Z = self.weights[i]@A + self.biases[i]
            self.Zs.append(Z)

            if i != len(self.weights)-1:
                self.activations.append(self.activation_f(Z))
            else:
                self.activations.append(self.activation_llf(Z))

        if Y is not None:
            return self._loss(self.activations[-1] , Y)
        return self.activations[-1]

    def backward_pass(self, X, Y, loss="cross entropy", activation="sigmoid", lla="softmax", forwardpass=False):
        self.loss_type_str = loss.lower()

        if forwardpass is False:
            self.forward_pass(X , Y , activation=activation , lla=lla)

        f , lf = self.activation_f , self.activation_llf

        if loss.lower() == "cross entropy":
            self.loss = self.Loss_CE
        elif loss.lower() in ["squared","mse","mean squared error"]:
            self.loss = self.Loss_sq

        AL = self.activations[-1]
        m = X.shape[1]

        deriv_map = {
            FCFFNN.sigmoid:FCFFNN.dSig,
            FCFFNN.tanh:FCFFNN.dtanh,
            FCFFNN.ReLU:FCFFNN.dReLU,
            FCFFNN.same:FCFFNN.dsame
        }

        if self.loss_type_str == "cross entropy" and lla.lower() == "softmax":
            dZ = AL - Y
        elif self.loss_type_str in ["squared","mse","mean squared error"] and lla.lower() in ["linear","same"]:
            dZ = (2/m)*(AL-Y)
        else:
            f_prime = deriv_map.get(lf , FCFFNN.dsame)
            dZ = (2/m)*(AL-Y)*f_prime(self.Zs[-1])

        self.dw , self.db = [] , []