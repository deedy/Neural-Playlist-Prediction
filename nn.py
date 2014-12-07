"""
Module for neural network models
for song similarity regression

Code mostly copied from http://deeplearning.net/tutorial/mlp.html

@author Siddharth Reddy <sgr45@cornell.edu>
12/3/14
"""

import os
import sys
import time

import numpy as np

import theano
import theano.tensor as T

from logistic_sgd import LogisticRegression

from math import log

class HiddenLayer(object):
  def __init__(self, rng, input, n_in, n_out, W=None, b=None, activation=T.tanh):
    self.input = input
   
    if W is None:
      W_values = np.asarray(
      rng.uniform(
        low=-np.sqrt(6. / (n_in + n_out)),
        high=np.sqrt(6. / (n_in + n_out)),
        size=(n_in, n_out)
      ),
      dtype=theano.config.floatX
      )
    if activation == theano.tensor.nnet.sigmoid:
      W_values *= 4

    self.activation = activation

    W = theano.shared(value=W_values, name='W', borrow=True)

    if b is None:
      b_values = np.zeros((n_out,), dtype=theano.config.floatX)
      b = theano.shared(value=b_values, name='b', borrow=True)

    self.W = W
    self.b = b

    lin_output = T.dot(input, self.W) + self.b
    self.output = (
    lin_output if activation is None
    else activation(lin_output)
    )
    self.params = [self.W, self.b]

  def run(self, input):
    output = T.dot(input, self.W) + self.b
    output = output if self.activation is None else self.activation(output)
    return output

class MLP(object):

  def __init__(self, rng, input, n_in, n_hidden, n_out):

    self.hiddenLayer = HiddenLayer(
    rng=rng,
    input=input,
    n_in=n_in,
    n_out=n_hidden,
    activation=T.tanh)

    self.logRegressionLayer = LogisticRegression(
    input=self.hiddenLayer.output,
    n_in=n_hidden,
    n_out=n_out)

    self.L2_sqr = (
    (self.hiddenLayer.W ** 2).sum()
    + (self.logRegressionLayer.W ** 2).sum())

    self.mean_squared_error = (self.logRegressionLayer.mean_squared_error)
    self.errors = self.logRegressionLayer.errors

    self.params = self.hiddenLayer.params + self.logRegressionLayer.params

  def run(self, input):
    hidden_output = self.hiddenLayer.run(input)
    logreg_output = self.logRegressionLayer.run(hidden_output)
    return logreg_output

class MultilayerPerceptron(object):

  def __init__(self):
    self.regressor = None

  def fit(self, X, Y, learning_rate=0.00001, L1_reg=0.00, L2_reg=0.0001, n_epochs=1,
       batch_size=1, n_hidden=500):

    train_set_x = np.asarray(X)
    train_set_y = np.asarray(Y)

    num_features = train_set_x.shape[1]

    n_train_batches = train_set_x.shape[0] / batch_size

    index = T.lscalar()
    x = T.matrix('x')
    y = T.vector('y')

    rng = np.random.RandomState(1234)

    regressor = MLP(
    rng=rng,
    input=x,
    n_in=num_features,
    n_hidden=n_hidden,
    n_out=1)

    cost = (
    regressor.mean_squared_error(y)
    + L2_reg * regressor.L2_sqr)

    gparams = [T.grad(cost, param) for param in regressor.params]

    updates = [
    (param, param - learning_rate * gparam)
    for param, gparam in zip(regressor.params, gparams)]

    train_model = theano.function(
      inputs=[x,y],
      outputs=cost,
      updates=updates,
      allow_input_downcast=True)

    for epoch in xrange(n_epochs):
      minibatch_avg_cost = 0
      for minibatch_idx in xrange(n_train_batches):
        tsx = train_set_x[(minibatch_idx * batch_size): ((minibatch_idx + 1) * batch_size),:]
        tsy = train_set_y[(minibatch_idx * batch_size): ((minibatch_idx + 1) * batch_size)]
        minibatch_avg_cost += train_model(tsx,tsy)
      print "%d\t%f" % (epoch, minibatch_avg_cost/n_train_batches)

    self.regressor = regressor

  def predict_log_proba(self, x):
    return log(self.regressor.run(x).eval())
