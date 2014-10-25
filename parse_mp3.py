from pydub import AudioSegment
import eyed3
import os.path
import numpy as np
import matplotlib.pyplot as plt

from IPython.core.debugger import Tracer


def parse_mp3(path):
  if not os.path.isfile(path):
    raise Exception('Could not locate {0}'.format(path))

  print('Decoding metadata... \n')
  id3tags = eyed3.load(path)
  print_song_metadata(id3tags)
  print_audio_metadata(id3tags)
  print('\n\n\n')

  print('Decoding audio (this may take a while) ... \n')
  audio = AudioSegment.from_mp3('data/duaa.mp3')
  print('Done decoding.\n')

  data = np.fromstring(audio._data, np.int16)

  channels = []
  for channel in xrange(audio.channels):
    channels.append(data[channel::audio.channels])
  Tracer()()

def print_song_metadata(id3tags):
  print('---------- SONG DETAILS ----------')
  print('Title:            {0}'.format(id3tags.tag.title))
  print('Artist:           {0}'.format(id3tags.tag.artist))
  print('Album:            {0}'.format(id3tags.tag.album))

def print_audio_metadata(id3tags):
  print('---------- AUDIO DETAILS ----------')
  print('Bitrate:          {0}'.format(id3tags.info.bit_rate_str))
  print('Mode:             {0}'.format(id3tags.info.mode))
  print('Sample Frequency: {0}'.format(id3tags.info.sample_freq))
  print('Size (in bytes):  {0}'.format(id3tags.info.size_bytes))
  print('Time (in s):      {0}'.format(id3tags.info.time_secs))
