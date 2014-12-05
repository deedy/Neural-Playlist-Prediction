"""
Module for generating synthetic data
@author Siddharth Reddy <sgr45@cornell.edu>
12/2/4
"""

from __future__ import division
from models import AudioFeatureSet
from random import random
from math import exp

import numpy as np
import uuid

def random_song_name():
    return str(uuid.uuid4())

def generate_synthetic_playsets(
    num_playsets=100, 
    num_songs_per_playset=25, 
    num_songs=100,
    num_features=50):
    """
    Generate random audio feature sets for songs
    (sample from uniform distribution)

    Generate random playsets by choosing seed songs,
    then iteratively adding songs with probability
    proportional to exp(-(L2 distance)) from the working
    center of the playset

    :param int num_playsets: Number of playsets
    :param int num_songs_per_playset: Playset length
    :param int num_songs: Number of unique songs in data set
    :param int num_features: Number of audio features
    :rtype tuple(list[set(str)],dict[str]=AudioFeatureSet)
    :return A list of playsets containing song IDs, and a dictionary
    that maps song IDs to AudioFeatureSet objects
    """

    assert num_songs_per_playset < num_songs, "Playset larger than set of songs!"

    # generate random audio feature sets for songs
    def random_afs():
        return np.random.random((num_features,))

    afshash = {random_song_name():random_afs() for _ in xrange(num_songs)}

    # generate random playsets
    def random_playset(n):
        songs = afshash.keys()
        def sample_from_songs(p=None):
            r = np.random.choice(range(len(songs)), replace=False, p=p)
            return (songs[:r]+songs[(r+1):], songs[r])
        songs, seed = sample_from_songs()
        ps = [seed]
        def compute_sample_probabilities(ps):
            ctr = sum(afshash[x] for x in ps)/len(ps)
            p = [exp(-np.linalg.norm(ctr-afshash[x])) for x in songs]
            z = sum(p)
            return [x/z for x in p]
        for _ in xrange(n):
            p = compute_sample_probabilities(ps)
            songs, new = sample_from_songs(p)
            ps.append(new)
        return ps

    playsets = [random_playset(num_songs_per_playset) for _ in xrange(num_playsets)]

    afshash = {k:AudioFeatureSet(v) for k, v in afshash.iteritems()}
    
    return (playsets, afshash)
