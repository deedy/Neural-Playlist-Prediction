"""
Module for playset model evaluation
@author Siddharth Reddy <sgr45@cornell.edu>
12/2/14
"""

from models import *
from datatools import munge_gtzan
from synth import generate_synthetic_playsets
from random import shuffle, sample

import pylab as plt

import sys
import math

def split_playsets(data, T=0.5):
    """
    Split playsets into training and test sets

    Each playset gets split in T (as opposed to splitting the 
    list of playsets in T and keeping individual playsets together)

    The test set will not contain any songs from the training set

    :param tuple(list[set(str)],dict[str]=AudioFeatureSet) data: 
    A list of playsets containing song IDs, and a dictionary
    that maps song IDs to AudioFeatureSet objects
    :param int T: Fraction of instances in training set
    (1 - fraction of instances in test set)
    :rtype tuple(train,test)
    :return A tuple of training and test data sets
    """

    playsets, afshash = data
    def split(x):
        shuffle(x)
        s = int(T*len(x))
        return (x[:s],x[s:])
    pre = [split(list(ps)) for ps in playsets]
    train = [x[0] for x in pre]
    test = [x[1] for x in pre]
    songs = lambda t: {s for ps in t for s in ps}
    pare = lambda t, h: {k:v for k, v in h.iteritems() if k in songs(t)}
    return ((train, pare(train, afshash)), (test, pare(test, afshash)))

def song_pairs_mse(model, data):
    """
    Compute mean-squared error of predicted conditional probabilities
    of song co-occurrence

    :param SongPairModel model: A trained song similarity regression model
    :param list[tuple(AudioFeatureSet,AudioFeatureSet,float)] data: 
    A list of (song A, song B, \hat{Pr}(A|B))
    :rtype float
    :return Mean-squared error
    """

    pred, act = [], []
    for d in data:
        x, y, p = d
        pred.append(math.exp(model.log_likelihood(x, y)))
        act.append(p)

    # BEGIN DEBUG
    #plt.scatter(pred, act)
    #plt.hist(pred)
    #plt.show()
    # END DEBUG

    mse = np.linalg.norm(np.asarray(pred) - np.asarray(act))

    return mse

def playsets_avg_ll(model, data):
    """
    Compute average log-likelihood of playsets

    :param PlaysetModel model: A trained playset model
    :param tuple(list[set(str)],dict[str]=AudioFeatureSet) data: 
    A list of playsets containing song IDs, and a dictionary
    that maps song IDs to AudioFeatureSet objects
    :rtype float
    :return Average log-likelihood
    """

    playsets, afshash = data
    avg_ll = sum(model.avg_log_likelihood((ps, afshash)) for ps in playsets) / len(playsets)
    return avg_ll

def evaluate_models(data, K=3):
    """
    Compute cross-validated average log-likelihood
    of held out playsets using models.PlaysetModel

    Compute cross-validated mean-squared error
    of held out song pairs using models.SongPairModel

    :param tuple(list[set(str)],dict[str]=AudioFeatureSet) data: 
    A list of playsets containing song IDs, and a dictionary
    that maps song IDs to AudioFeatureSet objects
    :param int K: K-fold cross-validation
    :rtype tuple(tuple(float,float),tuple(float,float))
    :return Average log-likelihood and mean-squared error
    of model vs. benchmark
    """
    
    model = PlaysetModel()
    bench_model = PlaysetModel(benchmark=True)
    bench_song_pair_model = BenchmarkSongPairModel()
    avg_ll = 0
    bench_avg_ll = 0
    mse = 0
    bench_mse = 0
    for _ in xrange(K):
        train, test = split_playsets(data)
        model.train(train)
        bench_model.train(train)
        avg_ll += playsets_avg_ll(model, test)
        bench_avg_ll += playsets_avg_ll(bench_model, test)

        test_song_pairs = get_song_pairs(test)
        mse += song_pairs_mse(model.song_pair_model, test_song_pairs)
        bench_mse += song_pairs_mse(bench_song_pair_model, test_song_pairs)

    return ((avg_ll/K), (bench_avg_ll/K), (mse/K), (bench_mse/K))

def main():
    #data_path = sys.argv[1]
    #data = munge_gtzan(data_path)
    data = generate_synthetic_playsets()

    # BEGIN DEBUG
    """
    cp = [x[2] for x in get_song_pairs(data)]
    plt.hist(cp)
    plt.show()
    """
    # END DEBUG

    avg_ll, bench_avg_ll, mse, bench_mse = evaluate_models(data)

    print "Playset Model\n-------------\nAverage log-likelihood: %f\nMean-squared error %f" % (avg_ll, mse)
    print ""
    print "Benchmark Model\n-------------\nAverage log-likelihood: %f\nMean-squared error %f" % (bench_avg_ll, bench_mse)

if __name__ == '__main__':
    main()
