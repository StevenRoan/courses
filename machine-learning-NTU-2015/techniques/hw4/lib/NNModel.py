import numpy as np


# In the formula. index L = # of layers (including input and output) - 1
# (e.g.2-5-1 nnm lenght=3 L=2)
# w^l, s^l 1<=l<L
# x^l => 0<=l<L, x^0 is input + bias


class NNModel:

    def __init__(self, trainData, neurons, wrange):
        self.matrix = trainData
        columnsNum = self.matrix.shape[1]
        self.trainData = self.matrix[:, range(0,   columnsNum - 1)]
        self.trainLabel = self.matrix[:,   columnsNum - 1]
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
                self.neurons[idx] + 1, self.neurons[idx + 1]) * length + self.l
            self.weights.append(w)
            # [0]=x^0, [1]=x^1, [2]=x^2,....[L-2]=x^{L-1}
            self.inputs.append(np.zeros((1, self.neurons[idx])))
            # [0]=s^1, [1]=s^2, [2]=s^3,....[L-2]=s^{L-2}
            self.signals.append(np.zeros((1, self.neurons[idx + 1])))
            # [0]=sigma^1 [1]=sigma^2
            self.errToSignalGradients.append(
                np.zeros((1, self.neurons[idx + 1])))

    # Train with backdrop propagation
    def train(self, step, T):
        for t in range(T):
            print '>>>>>>'
            xIdx = self._pickDataIdxRandomly()
            data = self.trainData[xIdx]
            self._updateForward(data)
            self._updateBackward(xIdx)
            self._updateWeights(xIdx, step)

    def test(self, testSet):
        print 'test'
        rowsNum = testSet.shape[0]
        columnsNum = testSet.shape[1]
        testLabel = testSet[:,  columnsNum - 1]
        testData = testSet[:, range(0,   columnsNum - 1)]
        bias = np.ones((rowsNum, 1))
        input = testData
        for idx in range(self.layersNum - 1):
            input = np.concatenate([bias, input], 1)
            s = input.dot(self.weights[idx])
            if (idx < self.layersNum - 2):
                input = self._tranfromNonLinear(s)
        # print (s)

    def _predict(self, data):
        self._updateForward(data)
        return 1 if (self.signals[self.layersNum - 2] > 0) else -1

    def _pickDataIdxRandomly(self):
        return np.random.randint(0, self.trainData.shape[0])

    def _updateForward(self, data):
        self.inputs[0] = data  # use column vector
        layersNum = self.layersNum
        for idx in range(layersNum - 1):
            biasedInput = np.concatenate([self.bias, self.inputs[idx]])
            self.signals[idx] = biasedInput.dot(self.weights[idx])
            # 1xm X mxn = 1xn, n is number of neurons (exclude bias term in the
            # next level)
            if (idx < layersNum - 2):
                self.inputs[
                    idx + 1] = self._tranfromNonLinear(self.signals[idx])

    def _updateBackward(self, xIdx):
        backBoundary = self.layersNum - 2
        for idx in range(backBoundary, -1, -1):
            print idx
            # When updating sigma, error to signal gradient, we don't need the
            # term from bias (+1) neuron
            if (idx == backBoundary):
                res = self.errToSignalGradients[
                    idx] = self._getRightMostLayerGradient(xIdx)
                print res
                print '---'
            else:
                sigma = self.errToSignalGradients[idx + 1]
                # remove the row of w related to bias term
                w = self.weights[idx + 1][1:len(self.weights[idx + 1]):1, :].T
                s = self.signals[idx]
                res = self.errToSignalGradients[idx] = sigma.dot(
                    w) * self._tranfromNonLinear(s, True)
                if self.a == True:
                    print 'oriW'
                    print self.weights[idx + 1]
                    print 'sigma'
                    print sigma
                    print 'w'
                    print w
                    print 's'
                    print s
                    print 'res'
                    print res
                    print '---'

    def _updateWeights(self, xIdx, step):
        print '*******'
        for idx in range(self.layersNum - 1):
            x = np.concatenate([self.bias, self.inputs[idx]])
            x = x.reshape(len(x), 1)
            sigma = self.errToSignalGradients[idx].T
            sigma = sigma.reshape(1, len(sigma))
            gradientM = x.dot(sigma)
            self.weights[idx] = self.weights[idx] - (step * gradientM)
            if self.b == True:
                print 'x'
                print(x)
                print 'sigma'
                print(sigma)
                print 'gradientM'
                print gradientM
                print 'new weights'
                print self.weights[idx]
                print '-----------'

    # Transform x of each layer to the signals
    def _tranfromNonLinear(self, x, derivative=False):
        if(derivative == False):
            return np.tanh(x)
        else:
            # derivative of tanh
            return 1 - np.square(np.tan(x))

    # xIdx is the picked train data
    # This is for square mean error
    def _getRightMostLayerGradient(self, xIdx):
        # if self.signals[len(self.signals) -1] \in 1x1, it is identical to
        # self.signals[len(self.signals) -1][0]
        # in right most case signal == input
        return -2 * (self.trainLabel[xIdx] - self.signals[len(self.signals) - 1])
