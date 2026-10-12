import numpy as np
import pickle
import json
import os
import sys
import time
import matplotlib.pyplot as plt

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
            return -np.sum(y*np.log(AL) + (1-y)*np.log(1-AL))/AL.shape[1]

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

        def back_propagate():
            nonlocal dZ
            df = deriv_map.get(self.activation_f , FCFFNN.dsame)

            for i in range(len(self.weights)-1 , -1 , -1):
                A_i = self.activations[-1]
                dW = (1/m)*dZ@A_i.T
                dB = (1/m)*np.sum(dZ , axis=1 , keepdims=True)
                self.db.append(dB) , self.dw.append(dW)

                if i > 0:
                    dZ = (self.weights[i].T@dZ)*df(self.Zs[i-1])
                else:
                    self.dZ_1 = dZ
            self.dw = self.dw[::-1]
            self.db = self.db[::-1]
        back_propagate()

    def Train_NN(self , Dataset=None , X1=None, Y1=None, k=1, iterations=1, epoch=1, lr=None, lam=0, loss="cross entropy", activation="sigmoid", lla="softmax", interval=100):
        Start = time.time()

        if lr is None:
            lr = FCFFNN.lr_constant

        if (X1 is None) and (Y1 is None) and (Dataset is not None):
            X_all = np.column_stack([item[0] for item in Dataset])
            Y_all = np.column_stack([item[1] for item in Dataset])
        elif (X1 is not None) and (Y1 is not None):
            X_all = X1
            Y_all = Y1
        else:
            print("WARNING : You must enter (X and Y) or the Datast, is none is entered the training will fail")
            return None

        num_samples = X_all.shape[1]
        self.losses_over_t = []

        for t in range(1,iterations+1):

            indices = np.random.choice(num_samples , size=k , replace=False)
            X = X_all[: , indices]
            Y = Y_all[: , indices]

            for mu in range(1,epoch+1):
                self.backward_pass(X,Y,loss=loss,activation=activation,lla=lla)

                for i in range(len(self.weights)):
                    self.weights[i] = (1-(lr(t)*lam))*self.weights[i] - (lr(t)*self.dw[i])
                    self.biases[i] = self.biases[i] - (lr(t)*self.db)
                self.losses_over_t.append(self.loss(self.activations[-1],Y))

                if t%interval==0:
                    print(f"At iteration {t} and epoch {mu} we have loss {self.losses_over_t[-1]}")

        return round(time.time()-Start , 3)

    def save_model(self , file):

        Data = {"W":self.weights , "B":self.biases , "L":self.layers}
        with open(file , "wb") as f:
            pickle.dump(Data , f)

    def load_model(self , file):
        
        with open(file , "rb") as f:
            Data = pickle.load(f)
            
        self.weights = Data["W"]
        self.biases = Data["B"]
        self.layers = Data["L"]

    def Test_NN(self , Test_ds=None , X=None , Y=None , limit=None , activation="sigmoid" , lla="softmax" , Sloss="cross entropy"):

        if (X is None) and (Y is None):
            X_all = np.column_stack([item[0] for item in Test_ds])
            Y_all = np.column_stack([item[1] for item in Test_ds])
        elif Test_ds is None:
            X_all = X
            Y_all = Y

        if Sloss.lower() == "cross entropy":
            self.loss = self.Loss_CE
        elif Sloss.lower() in ["squared", "mse", "mean squared error"]:
            self.loss = self.Loss_sq

        self.test_loss = []
        if limit is not None:
            X_all = X_all[: , limit]
            Y_all = Y_all[: , limit]

        for i in range(X_all.shape[1]):
            x = X_all[: , [i]]
            y = Y_all[: , [i]]
            self.forward_pass(x , activation=activation , lla=lla , Sloss=Sloss)
            self.test_loss.append(self.loss(self.activations[-1] , y))
            if i%1000 == 0:
                print(f"At the {i} element the loss is {self.test_loss[-1]}")
        return self.test_loss

    def graph(self , show=True , test_loss=False , train_loss=False):
        if train_loss:
            Y = self.losses_over_t
            plt.plot(Y , color="blue" , label="Train Loss")

        if test_loss:
            y = self.test_loss
            plt.plot(y , color="red" , label="Test Loss")

        plt.ylabel("Loss")
        plt.xlabel("Each iteration ove each Batch")
        plt.grid(True)
        plt.legend()

        if show is True:
            plt.show()

if __name__ == "__main__":

    pass