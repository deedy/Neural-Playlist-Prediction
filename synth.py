"""
Module for generating synthetic data
@author Siddharth Reddy <sgr45@cornell.edu>
12/2/4
"""

def generate_synthetic_playsets(
    num_playsets=10, 
    num_songs_per_playset=10, 
    num_songs=15):
    """
    Generate random playsets 

    Use a multivariate Gaussian to generate random 
    audio feature sets for songs 

    Each playset has a center

    Covariance between two centers/playsets is a function
    of # of shared songs


    """
    raise NotImplementedError