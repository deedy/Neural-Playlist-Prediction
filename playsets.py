"""
Module for automatic playset generation
@author Siddharth Reddy <sgr45@cornell.edu>
12/5/14
"""

from random import random

def make_playset(seed, pool, length=10):
    """
    :param set(str) seed: 
    :param dict[str]=AudioFeatureSet pool: 
    :param int length: 
    :rtype set(str)
    :return 
    """

    assert length <= len(pool), "Playset larger than song pool!"

    raise NotImplementedError

def get_gtzan_song_pool(data_path):
    raise NotImplementedError

def get_tpb_song_pool(data_path):
    raise NotImplementedError


def main():
    data_path = sys.argv[1]
    pool = get_gtzan_song_pool(data_path)
    seed = set(pool.keys()[random(len(pool))])
    ps = make_playset(seed, pool)

    print "Seed:"
    print seed
    print "Generated playset:"
    print ps

if __name__ == '__main__':
    main()