import numpy as np
from lib.NNModel import NNModel
string = open("data/hw4-q11_nnet_train.dat")
train = np.loadtxt(string)
string = open("data/hw4-q11_nnet_test.dat")
testData = np.loadtxt(string)
# M = [1, 4, 11, 16, 21]

NNM = NNModel(train, [2, 5, 3, 1], [-0.1, 0.1])
NNM.a = False
NNM.a = True
NNM.b = True
# NNM.b = False
NNM.train(1,8)
NNM.test(testData)
print 'done'

# single layer neural network