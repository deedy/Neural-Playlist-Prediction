"""
Module for logistic regression layer of nnet

Code mostly copied from http://deeplearning.net/tutorial/logreg.html

@author Siddharth Reddy <sgr45@cornell.edu>
12/6/14
"""

import cPickle
import gzip
import os
import sys
import time

import numpy

import theano
import theano.tensor as T


class LogisticRegression(object):

  def __init__(self, input, n_in, n_out):
    self.W = theano.shared(
      value=numpy.zeros(
        (n_in, n_out),
        dtype=theano.config.floatX
      ),
      name='W',
      borrow=True
    )
    self.b = theano.shared(
      value=numpy.zeros(
        (n_out,),
        dtype=theano.config.floatX
      ),
      name='b',
      borrow=True
    )

    #self.y_pred = T.nnet.softmax(T.dot(input, self.W) + self.b)
    self.y_pred = 1 / (1 + T.exp(-(T.dot(input, self.W) + self.b)))

    self.params = [self.W, self.b]

  def mean_squared_error(self, y):
    return T.mean(T.pow(self.y_pred - y, 2))
  
  def run(self, input):
    return 1 / (1 + T.exp(-(T.dot(input, self.W) + self.b)))

  def errors(self, y):
    if y.ndim != self.y_pred.ndim:
      raise TypeError(
        'y should have the same shape as self.y_pred',
        ('y', y.type, 'y_pred', self.y_pred.type)
      )
    if y.dtype.startswith('int'):
      return T.mean(T.pow(self.y_pred - y, 2))
    else:
      raise NotImplementedError()
