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

            