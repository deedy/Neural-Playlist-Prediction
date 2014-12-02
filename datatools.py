"""
Module for playset model data munging
@author Siddharth Reddy <sgr45@cornell.edu>
12/2/14
"""

from models import AudioFeatureSet
from main import run

import cPickle as pickle
import os

DATA_DIR = 'data'

def preproc(path):
    """
    Run main.py recursively on files in directory    

    :param str path: Path to data directory
    """

    files = os.listdir(path)
    for f in files:
        f = os.path.join(path, f)
        if os.path.isfile(f):
            if f[-3:]=='.au' or f[-4:]=='.mp3':
                run(f)
        else:
            preproc(f)

def song_name_from_file(f):
    return f.split('.pik')[0]

def munge_gtzan(path):
    """
    Parse preprocessed GTZAN files to model-ready data set

    :param str path: Path to preprocessed output directory

    Directory should be structured as follows:
    path/genres
    path/genres/name_of_genre
    path/genres/name_of_genre/name_of_song.pik

    :rtype tuple(list[set(str)],dict[str]=AudioBite)
    :return A list of playsets containing song IDs, and a dictionary
    that maps song IDs to AudioBite objects
    """
    
    playsets = []
    afshash = {}
    playset_dirs = os.listdir(os.path.join(path, 'genres'))
    for playset_dir in playset_dirs:
        songs = [f for f in os.listdir(os.path.join(path, 'genres', playset_dir)) if f[-4:]=='.pik']
        playsets.append({song_name_from_file(f) for f in songs})
        for f in songs:
            snf = song_name_from_file(f)
            if snf not in afshash:
                afshash[snf] = AudioFeatureSet(pickle.load(open(os.path.join(path, 'genres', playset_dir, f), 'rb')))

    return (playsets, afshash)

def munge_playsets(path_to_playsets, path_to_ab):
    """
    Parse preprocessed GTZAN files to model-ready data set

    :param str path_to_playsets: Path to playsets object file

    Should be a pickled object with type list[set(str)]

    :param str path_to_ab: Path to AudioBite objects

    Should be a directory containing pickled objects with
    type AudioBite and filename=songID.pik

    :rtype tuple(list[set(str)],dict[str]=AudioBite)
    :return A list of playsets containing song IDs, and a dictionary
    that maps song IDs to AudioBite objects
    """
    raise NotImplementedError

def main():
    preproc(DATA_DIR)

if __name__ == '__main__':
    main()