from pydub import AudioSegment
import eyed3
import os.path
import numpy as np
import time

import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import matplotlib.cm as cm
# from matplotlib.colors import LogNorm

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
  decode_start_time = time.time()
  audio = AudioSegment.from_file(path)
  print('Done decoding in {0} seconds to decode {1:.2f} seconds of audio.\n'.format(time.time() - decode_start_time, len(audio)))

  # Get last 30 seconds
  audio = audio[-40000:-10000]

  # np.int32 or np.int16?
  data = np.fromstring(audio._data, np.int32)

  # What is the correct way to separate data into channels?
  # channels = []
  # for channel in xrange(audio.channels):
  #   channels.append(data[channel::audio.channels])

  Pxx, freqs, bins = get_spectrogram(data, plot=True)
  Tracer()()

  # Maybe do a 3D visual?
  # from mpl_toolkits.mplot3d import Axes3D
  # fig = plt.figure()
  # ax = fig.gca(projection='3d')
  # B, F = meshgrid(bins, freqs)
  # ax.plot_surface(B, F, Pxx, cstride=10, rstride=10, cmap=cm.coolwarm)
  # show()

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

DEFAULT_WINDOW_SIZE = 4096
DEFAULT_OVERLAP_RATIO = 0.5
DEFAULT_SAMPLE_FREQUENCY = 44100

def get_spectrogram(data,
    nfft = DEFAULT_WINDOW_SIZE,
    overlap = DEFAULT_OVERLAP_RATIO,
    fs = DEFAULT_SAMPLE_FREQUENCY,
    plot = False):
  # matplotlib.pyplot and matplotlib have different specgram implementations -
  # pyplot plots, and mlab merely calculates.
  if plot:
    ax = plt.subplot(111)
    Pxx, freqs, bins, im = ax.specgram(data, NFFT = nfft, Fs = fs, noverlap = (int(nfft*overlap)))
    plt.xlabel('Time (in seconds)')
    plt.ylabel('Frequency (in Hz)')
    plt.xlim(0, len(data)/fs)
    plt.ylim(0, fs//2) # Nyquist sampling
    plt.show()
    return Pxx, freqs, bins
  Pxx, freqs, bins = mlab.specgram(data, NFFT = nfft, Fs = fs, noverlap = (int(nfft*overlap)))
  return Pxx, freqs, bins


