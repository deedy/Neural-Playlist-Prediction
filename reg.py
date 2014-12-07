"""
Module for logistic regression with Theano
@author Siddharth Reddy <sgr45@cornell.edu>
12/4/14
"""

from __future__ import division
from math import log, exp, tanh

import theano
import theano.tensor as T

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

import numpy as np

class LogisticRegression(object):

  def __init__(self):
    self.w = None
    self.b = None

  def fit(self, X, Y, lrate=0.01, training_steps=100, coeff_reg=0.1):#, batch_size=1):
    """
    Batch gradient descent for maximum-likelihood estimation
    """
    num_instances = len(X)
    num_features = len(X[0])

    x = T.matrix("x")
    y = T.vector("y")
    w = theano.shared(np.random.random((num_features,)), name="w")
    b = theano.shared(0., name="b")

    #p = 1 / (1 + T.exp(-(T.dot(x, w) + b)))
    p = (1 + T.tanh(-(T.dot(x, w) + b))) / 2
    cost = ((p - y) ** 2).sum() + coeff_reg * (w ** 2).sum()
    gw, gb = T.grad(cost, [w, b])

    train = theano.function(
      inputs=[x,y],
      outputs=[cost],
      updates=((w, w - lrate * gw), (b, b - lrate * gb)),
      allow_input_downcast=True)

    cst = [0] * training_steps
    #num_batches = num_instances // batch_size
    for i in xrange(training_steps):
      """
      for j in xrange(num_batches):
        #lidx = j*batch_size
        #uidx = min(num_instances, lidx+batch_size)
        #err = train(X[lidx:uidx], Y[lidx:uidx])
        err = train(X[j:(j+1)], Y[j:(j+1)])
      """
      err = train(X, Y)
      cst[i] = sum(err)
      print "%d\t%f" % (i, cst[i])

    plt.plot(cst)
    #plt.show()
    plt.savefig('bgd.png')

    self.w = w.get_value()
    self.b = b.get_value()

  def predict_log_proba(self, x, eps=1e-6):
    #return -log(1 + exp(-(np.dot(x, self.w) + self.b)))
    p = (1 + tanh(-(np.dot(x, self.w) + self.b))) / 2
    p = max(p, eps)
    return log(p)
