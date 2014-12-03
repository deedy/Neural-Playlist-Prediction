"""
Module for playset model and song similarity
regression model
@author Siddharth Reddy <sgr45@cornell.edu>
12/2/14
"""

from __future__ import division
from AudioBite import AudioBite
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from math import log
from random import random

import numpy as np

import time

class AudioFeatureSet(object):
    """
    Class for mapping mel-spectrograms to clean,
    standardized audio feature sets
    """

    def __init__(self, x):
        """
        Initialize audio feature set object

        :param AudioBite|np.ndarray x: 
        """
    
        self.vec = x.mel_specgram.flatten() if isinstance(x, AudioBite) else x

    def get(self):
        return self.vec

class SongPairModel(object):
    """
    Class for regression model that maps two audio feature
    sets to a probability of co-occurrence in a playset

    #Uses Kernelized Support Vector Regression
    Uses Logistic Regression with L2 regularization
    """

    def __init__(self):
        self.model = None

    def join_feature_sets(self, x, y):
        """
        Concatenate two song audio feature sets

        :param AudioFeatureSet x: An audio feature set of a song
        :param AudioFeatureSet y: An audio feature set of a song
        :rtype np.ndarray
        :return Feature set of x :: feature set of y
        """

        #return np.concatenate((x.get(), y.get()))
        return (x.get()-y.get())**2 # DEBUG

    def train(self, data):
        """
        Train regression model on a training data set

        :param list[tuple(AudioFeatureSet,AudioFeatureSet,float)] data: 
        A list of (song A, song B, \hat{Pr}(A|B))
        """
        
        X = [self.join_feature_sets(e[0], e[1]) for e in data]
        Y = [e[2] for e in data]

        print "Training model on data set with %d instances and %d features..." % (len(X), len(X[0]))
        start_time = time.time()
        self.model = LogisticRegression()
        #self.model = SVC(kernel='rbf', C=1.0, probability=True)
        self.model.fit(X, Y)
        print "Done in %f seconds." % (time.time() - start_time)

    def log_likelihood(self, x, y):
        """
        Compute log likelihood of co-occurrence Pr(X|Y)

        :param AudioFeatureSet x: An audio feature set of a song
        :param AudioFeatureSet y: An audio feature set of a song
        :rtype float
        :return A real value between 0 and 1
        """

        f = self.join_feature_sets(x, y)
        return self.model.predict_log_proba(f)[0,0]

class BenchmarkSongPairModel(object):
    def __init__(self):
        pass

    def train(self, data):
        pass

    def log_likelihood(self, x, y):
        r = random()
        return log(r)

class PlaysetModel(object):
    """
    Class for generative model of playsets
    """

    def __init__(self, benchmark=False):
        self.song_pair_model = None
        self.benchmark = benchmark 

    def train(self, data):
        """
        Estimate generative model using a training data set

        :param tuple(list[set(str)],dict[str]=AudioFeatureSet) data: 
        A list of playsets containing song IDs, and a dictionary
        that maps song IDs to AudioFeatureSet objects
        """

        pairs = get_song_pairs(data)

        self.song_pair_model = BenchmarkSongPairModel() if self.benchmark else SongPairModel()
        self.song_pair_model.train(pairs)

    def avg_log_likelihood(self, x):
        """
        Compute the average log-likelihood of observing a playset
        under the current model

        :param tuple(set(str),dict[str]=AudioFeatureSet) x:
        A playset containing song IDs, and a dictionary
        that maps song IDs to AudioFeatureSet objects 
        :rtype float
        :return Log-likelihood of playset
        """

        ps, afshash = x
        ps = list(ps)
        avg_ll = sum(
            self.song_pair_model.log_likelihood(
                afshash[a], afshash[b]) for i, a in enumerate(
                ps) for b in ps[(i+1):]) / (len(ps)*(len(ps)-1)/2)

        return avg_ll

def get_song_pairs(data):
    """
    Generate song pairs and estimated conditional 
    probabilities of co-occurrence from a list
    of playsets

    :param list[set(str)] data: A list of playsets
    :rtype list[tuple(AudioFeatureSet,AudioFeatureSet,float)] 
    :return A list of (song A, song B, \hat{Pr}(A|B))
    """
    
    playsets, afshash = data
    songs = {x for ps in playsets for x in ps}
    P = {x:0 for x in songs}
    CP = {(x,y):0 for x in songs for y in songs if x!=y}

    for ps in playsets:
        ps = list(ps)
        for i, x in enumerate(ps):
            P[x] += 1
            for y in ps[(i+1):]:
                CP[(x,y)] += 1

    num_songs = len(songs)
    CP = {k:(v/P[k[1]]) for k, v in CP.iteritems()}

    return [(afshash[x],afshash[y],CP[(x,y)]) for x in songs for y in songs if x!=y]

