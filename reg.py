"""
Module for logistic regression with Theano
@author Siddharth Reddy <sgr45@cornell.edu>
12/4/14
"""

from math import log, exp

import theano
import theano.tensor as T

import matplotlib.pyplot as plt

import numpy as np

class LogisticRegression(object):

    def __init__(self):
        self.w = None
        self.b = None

    def fit(self, X, Y, lrate=0.001, training_steps=300):
        num_features = len(X[0])

        x = T.matrix("x")
        y = T.vector("y")
        w = theano.shared(np.random.random((num_features,)), name="w")
        b = theano.shared(0., name="b")

        p = 1 / (1 + T.exp(-T.dot(x, w) - b))
        cost = ((p - y) ** 2).sum() + (w ** 2).sum()
        gw, gb = T.grad(cost, [w, b])

        train = theano.function(
            inputs=[x,y],
            outputs=[p, cost],
            updates=((w, w - lrate * gw), (b, b - lrate * gb)))

        cst = []
        for _ in xrange(training_steps):
            pred, err = train(X, Y)
            cst.append(err)

        plt.plot(cst)
        plt.show()

        self.w = w.get_value()
        self.b = b.get_value()

    def predict_log_proba(self, x):
        return -log(1 + exp(-(np.dot(x, self.w) + self.b)))