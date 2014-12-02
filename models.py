"""
Module for playset model and song similarity
regression model
@author Siddharth Reddy <sgr45@cornell.edu>
12/2/14
"""

from __future__ import division
from sklearn.svm import SVR
from sklearn.linear_model import LogisticRegression
from collections import defaultdict
from math import log

import numpy as np

class AudioFeatureSet(object):
    """
    Class for mapping mel-spectrograms to clean,
    standardized audio feature sets
    """

    def __init__(self, ab):
        """
        Initialize audio feature set object

        :param AudioBite ab: An AudioBite object
        """
    
        self.vec = ab.mel_specgram.flatten()

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

        return np.concatenate(x.get(), y.get())

    def train(self, data):
        """
        Train regression model on a training data set

        :param list[tuple(AudioFeatureSet,AudioFeatureSet,float)] data: 
        A list of (song A, song B, \hat{Pr}(A|B))
        """
        
        X = [self.join_feature_sets(e[0], e[1]) for e in data]
        Y = [e[2] for e in data]

        self.model = LogisticRegression()
        self.model.fit(X, Y)

    def log_likelihood(self, x, y):
        """
        Compute log likelihood of co-occurrence Pr(X|Y)

        :param AudioFeatureSet x: An audio feature set of a song
        :param AudioFeatureSet y: An audio feature set of a song
        :rtype float
        :return A real value between 0 and 1
        """

        f = self.join_feature_sets(x, y)
        return self.model.predict_log_proba(f)

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
    songs = {x for x in ps for ps in playsets}
    P = defaultdict(int)
    CP = defaultdict(int)

    for ps in playsets:
        ps = list(ps)
        for i, x in enumerate(ps):
            P[x] += 1
            for y in ps[(i+1):]:
                P[(x,y)] += 1

    P = {k:(v/len(playsets)) for k, v in P.iteritems()}
    CP = {k:(v/P[k[1]]) for k, v in CP.iteritems()}

    return [(AudioFeatureSet(
        afshash[x]),AudioFeatureSet(
        afshash[y]),CP[(x,y)]) for x in songs for y in songs if x!=y]

class PlaysetModel(object):
    """
    Class for generative model of playsets
    """

    def __init__(self):
        self.song_pair_model = None    

    def train(self, data):
        """
        Estimate generative model using a training data set

        :param tuple(list[set(str)],dict[str]=AudioBite) data: 
        A list of playsets containing song IDs, and a dictionary
        that maps song IDs to AudioBite objects
        """

        pairs = get_song_pairs(data)

        self.song_pair_model = SongPairModel()
        self.song_pair_model.train(pairs)

    def avg_log_likelihood(self, x):
        """
        Compute the average log-likelihood of observing a playset
        under the current model

        :param tuple(set(str),dict[str]=AudioBite) x:
        A playset containing song IDs, and a dictionary
        that maps song IDs to AudioBite objects 
        :rtype float
        :return Log-likelihood of playset
        """

        ps, afshash = x
        ps = list(ps)
        avg_ll = sum(
            self.song_pair_model.log_likelihood(
                afshash[a], afshash[b]) for i, a in enumerate(
                ps) for b in ps[(i+1):]) / (len(ps)**2)

        return avg_ll
