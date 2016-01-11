import numpy as np
from lib.NNModel import NNModel
string = open("data/hw4-q11_nnet_train.dat")
train = np.loadtxt(string)
# string = open("data/hw4-q11_nnet_test.dat")
# test = np.loadtxt(string)
# M = [1, 4, 11, 16, 21]

NNM = NNModel(train, [2, 2, 3, 1], [-0.1, 0.1])
NNM.train(1,1)
# single layer neural network