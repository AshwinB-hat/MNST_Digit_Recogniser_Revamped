import struct
import os
import scipy.io
import numpy as np


class MultivariateLogisticRegression:
    def __init__(self, it=10000, lr=0.01, verbose=False):
        self.lr = lr
        self.it = it
        self.verbose = verbose

    def __sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def __loss(self, h, y):
        return (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()

    def fit(self, X, y):
        # weight initialisation
        self.theta = np.zeros((X.shape[1], y.shape[1]))

        for i in range(self.it):

            z = np.dot(X, self.theta)
            h = self.__sigmoid(z)
            gradient = np.dot(X.T, (h - y)) / y.shape[0]

            self.theta -= self.lr * gradient

            if (self.verbose == True and i % 1000 == 0):
                z = np.dot(X, self.theta)
                h = self.__sigmoid(z)
                print(f'loss: {self.__loss(h,y)}\t')

    def predict(self, X):
        return np.reshape(np.argmax(self.__sigmoid(np.dot(X, self.theta)), axis=1) + 1, (X.shape[0], 1))

    def accuracy(self,X,Y):
        return (Y==self.predict(X)).mean()


def main():
    mat = scipy.io.loadmat('ex3data1.mat')
    X=mat['X']
    Y=mat['y']

    #initialisation of training labels
    y = np.zeros((X.shape[0], 10), dtype=int)
    for i in range(Y.shape[0]):
        y[i][Y[i] - 1] = int(1)

    model = MultivariateLogisticRegression(it=5000, lr=0.1)
    model.fit(X,y)
    print(model.accuracy(X,Y))

if __name__=="__main__":
    main()

