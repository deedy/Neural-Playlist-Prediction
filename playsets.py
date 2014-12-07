"""
Module for automatic playset generation
@author Siddharth Reddy <sgr45@cornell.edu>
12/5/14
"""

from synth import generate_synthetic_playsets
from datatools import munge_gtzan, munge_tpb
from models import PlaysetModel

from random import random
from math import exp
import sys

import numpy as np

def grow_playset(model, x, pool):
  """
  Sample a new song to add to a playset

  :param PlaysetModel model: A trained generative model
  of playsets
  :param tuple(set(str), dict[str]=AudioFeatureSet) x: A playset
  :param dict[str]=AudioFeatureSet pool: A pool of possible songs
  to add to the playset
  :rtype tuple(set(str), dict[str]=AudioFeatureSet)
  :return The playset x, with one new song
  """

  ps, afshash = x
  p = [exp(model.log_cond_prob(x, (y, afs))) for y, afs in pool.iteritems()]
  z = sum(p)
  p = [e/z for e in p]
  y = np.random.choice(pool.keys(), p=p)
  afshash[y] = pool[y]
  del pool[y]
  return (ps|{y}, afshash), pool

def make_playset(model, seed, pool, length=10):
  """
  Sample a playset

  :param PlaysetModel model: A trained generative model
  of playsets
  :param tuple(set(str), dict[str]=AudioFeatureSet) seed: A playset seed 
  (should be non-empty), usually provided
  by the user
  :param dict[str]=AudioFeatureSet pool: A pool of possible songs
  to include in the playset
  :param int length: Desired playset length
  :rtype set(str)
  :return A playset sampled using a trained
  generative model
  """

  ps, afshash = seed
  assert length <= len(pool), "Playset larger than song pool!"
  assert length >= len(ps), "Playset is already larger than desired length!"

  for _ in xrange(length-len(ps)):
    seed, pool = grow_playset(model, seed, pool)

  return seed

def main():
  data_path = sys.argv[1]
  data = munge_gtzan(data_path)
  #data = generate_synthetic_playsets()
  pool = data[1]
  seed_song = np.random.choice(pool.keys())
  seed_afshash = {seed_song:pool[seed_song]}
  seed = (set([seed_song]), seed_afshash)
  model = PlaysetModel()#benchmark=True)
  model.train(data)
  ps = make_playset(model, seed, pool)

  print "Seed: %s" % (seed_song)
  print "Generated playset:"
  print '\n'.join(ps[0])

if __name__ == '__main__':
  main()