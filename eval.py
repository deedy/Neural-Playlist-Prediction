"""
Module for playset model evaluation
@author Siddharth Reddy <sgr45@cornell.edu>
12/2/14
"""

from models import PlaysetModel, SongPairModel, get_song_pairs
from random import shuffle

import sys

def traintestsplit(data, T):
    """
    Split data into training and test sets

    :param int T: Fraction of instances in training set
    (1 - fraction of instances in test set)
    :rtype tuple(train,test)
    :return A tuple of training and test data sets
    """

    s = int(T*len(playsets))
    train = playsets[:s]
    test = playsets[s:]
    return (train, test)

def split_playsets(data, T=0.5):
    """
    Split playsets into training and test sets

    The test set will not contain any songs from the training set

    :param tuple(list[set(str)],dict[str]=AudioBite) data: 
    A list of playsets containing song IDs, and a dictionary
    that maps song IDs to AudioBite objects
    :param int T: Fraction of instances in training set
    (1 - fraction of instances in test set)
    :rtype tuple(train,test)
    :return A tuple of training and test data sets
    """

    playsets, afshash = data
    shuffle(playsets)
    train, test = traintestsplit(T)
    songs = lambda t: {s for s in ps for ps in t}
    pare = lambda t, h: {k:v for k, v in h.iteritems() if k in songs(t)}
    return ((train, pare(train, afshash)), (test, pare(test, afshash)))
    
def split_song_pairs(data, T=0.5):
    """
    Split song pairs into training and test sets

    The test set will not contain any songs from the training set

    :param tuple(list[set(str)],dict[str]=AudioBite) data: 
    A list of playsets containing song IDs, and a dictionary
    that maps song IDs to AudioBite objects
    :param int T: Fraction of instances in training set
    (1 - fraction of instances in test set)
    :rtype tuple(train,test)
    :return A tuple of training and test data sets
    """

    shuffle(data)
    train, test = traintestsplit(T)
    return (train, test)

def song_pair_mse(model, data):
    """
    Compute mean-squared error of predicted conditional probabilities
    of song co-occurrence

    :param SongPairModel model: A trained song similarity regression model
    :param list[tuple(AudioFeatureSet,AudioFeatureSet,float)] data: 
    A list of (song A, song B, \hat{Pr}(A|B))
    :rtype float
    :return Mean-squared error
    """

    mse = 0
    for d in data:
        x, y, p = d
        mse += (model.log_likelihood(x, y) - p) ** 2
    return mse / len(data)

def playset_avg_ll(model, data):
    """
    Compute average log-likelihood of playsets

    :param PlaysetModel model: A trained playset model
    :param tuple(list[set(str)],dict[str]=AudioBite) data: 
    A list of playsets containing song IDs, and a dictionary
    that maps song IDs to AudioBite objects
    :rtype float
    :return Average log-likelihood
    """

    playsets, afshash = data
    avg_ll = sum(model.avg_log_likelihood((ps, afshash)) for ps in playsets) / len(playsets)
    return avg_ll

def evaluate_playset_model(data, K=3):
    """
    Compute cross-validated average log-likelihood
    of held out playsets using models.PlaysetModel

    :param tuple(list[set(str)],dict[str]=AudioBite) data: 
    A list of playsets containing song IDs, and a dictionary
    that maps song IDs to AudioBite objects
    :param int K: K-fold cross-validation
    :rtype float
    :return Average log-likelihood
    """
    
    model = PlaysetModel()
    cv_avg_ll = 0
    for _ in xrange(K):
        train, test = split_playsets(data)
        model.train(train)
        cv_avg_ll += playset_avg_ll(model, test)

    return cv_avg_ll / K

def evaluate_song_pair_model(data, K=3):
    """
    Compute cross-validated mean-squared error
    of held out song pairs using models.SongPairModel

    :param tuple(list[set(str)],dict[str]=AudioBite) data: 
    A list of playsets containing song IDs, and a dictionary
    that maps song IDs to AudioBite objects
    :param int K: K-fold cross-validation
    :rtype float
    :return Mean-squared error
    """
    
    song_pairs = get_song_pairs(data)

    model = PlaysetModel()
    cv_mse = 0
    for _ in xrange(K):
        train, test = split_song_pairs(song_pairs)
        model.train(train)
        cv_mse += song_pair_mse(model, test)

    return cv_mse / K

def main():
    data_path = sys.argv[1]
    data = datatools.munge_gtzan(data_path)
    cv_avg_ll, cv_mse = evaluate(data)

    print "Average log-likelihood: %f\nMean-squared error %f" % (cv_avg_ll, cv_mse)

if __name__ == '__main__':
    main()