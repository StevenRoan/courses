import numpy as np


# In the formula. index L = # of layers (including input and output) - 1 (e.g.2-5-1 nnm lenght=3 L=2)
# w^l, s^l 1<=l<L
# x^l => 0<=l<L, x^0 is input + bias


class NNModel:

    def __init__(self, trainData, neurons, wrange):
        self.matrix = trainData
        rowsNum = self.matrix.shape[1]
        self.trainData = self.matrix[:, range(0,  rowsNum - 1)]
        self.trainLabel = self.matrix[:,  rowsNum - 1]
        # array of numbers that idicate the neuron of each
        # d-nerouns[0]-neurons[1]..-1
        self.neurons = neurons
        self.layersNum = len(self.neurons)
        self.l = wrange[0]
        self.r = wrange[1]
        self.bias = np.array([1])
        self._instantiateW()

    def _instantiateW(self):
        length = self.r - self.l
        self.weights = []
        self.signals = []  # notation s in the handout
        self.inputs = []  # notation x in the handout
        self.errToSignalGradients = []  # notation of sigmas in the handout

        for idx in range(self.layersNum - 1):
            # self.neurons[idx]+1 is for bias term
            w = np.random.rand(
                self.neurons[idx+1], self.neurons[idx]+1) * length + self.l
            self.weights.append(w)
            # [0]=s^1, [1]=s^2, [2]=s^3,....[L-1]=s^{L-2}
            self.signals.append(np.zeros((self.neurons[idx + 1], 1))) 

            # [0]=sigma^1 [1]=sigma^2
            self.errToSignalGradients.append(
                np.zeros((self.neurons[idx + 1], 1)))

            # [0]=x^0, [1]=x^1, [2]=x^2,....[L-1]=x^{L-1}
            self.inputs.append(np.zeros((self.neurons[idx], 1)))

        # Last element of neroun won't be push to the inputs array

    # Train with backdrop propagation
    def train(self, step, T):
        # Forward
        for t in range(T):
            xIdx = self._pickDataIdxRandomly()
            self._updateForward(xIdx)
            self._updateBackward(xIdx)

    def _pickDataIdxRandomly(self):
        return np.random.randint(0, self.trainData.shape[0])

    def _updateForward(self, xIdx):
        self.inputs[0] = self.trainData[xIdx].T # use column vector
        layersNum = self.layersNum
        for idx in range(layersNum-1):
            tmp = np.concatenate([self.bias, self.inputs[idx]])
            self.signals[idx] = self.weights[idx].dot(tmp)
            # 1xm X mxn = 1xn, n is number of neurons (exclude bias term in the
            # next level)
            if (idx < layersNum-2):
                self.inputs[idx + 1] = self._tranfromNonLinear(self.signals[idx])

    def _updateBackward(self, xIdx):
        backBoundary = self.layersNum - 2
        for idx in range(backBoundary, 1, -1):
            if (idx == backBoundary):
                self.errToSignalGradients[
                    idx] = self._getRightMostLayerGradient(xIdx)
            else:
                self.errToSignalGradients[idx] = self.errToSignalGradients[(
                  idx)].T.dot(self.weights[idx]) * self._tranfromNonLinear(self.signals[idx], True)

    # Transform x of each layer to the signals
    def _tranfromNonLinear(self, x, derivative=False):
        if(derivative == False):
            return np.tanh(x)
        else:
            # derivative of tanh
            return 1 - np.tan(x) ^ 2

    # xIdx is the picked train data
    # This is for square mean error
    def _getRightMostLayerGradient(self, xIdx):
        # if self.signals[len(self.signals) -1] \in 1x1, it is identical to
        # self.signals[len(self.signals) -1][0]
        return -2 * (self.trainLabel[xIdx] - self.signals[len(self.signals) - 1]) # in right most case signal == input
