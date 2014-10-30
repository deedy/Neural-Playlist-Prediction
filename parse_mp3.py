from pydub import AudioSegment # Reading Audio data
import eyed3 # Reading Audio tags
import subprocess as sbp # Used to run ffmpeg on command line
import os
import numpy as np
import time
import sys

from IPython.core.debugger import Tracer

AUDIO_SAMPLE_START_S = 30
AUDIO_SAMPLE_END_S = 60

# Standard form - Constant Bitrate 192kbps, Sample rate 44100 Hz
STANDARD_FREQUENCY = 44100
STANDARD_BITRATE = 192

def parse_mp3(path):
  print('-------------------------------------------')
  print('   BEGINNING TO PARSE {0}'.format(os.path.basename(path)))
  print('-------------------------------------------\n')

  print('Decoding metadata... \n')
  id3tags = eyed3.load(path)
  print_song_metadata(id3tags)
  print_audio_metadata(id3tags)
  print('\n\n\n')

  print('Ensuring file is in the correct standard format...\n')
  path = convert_mp3_to_standard_form(path, id3tags)

  print('Decoding audio... ')
  decode_start_time = time.time()
  audio = AudioSegment.from_file(path)
  print('Done decoding in {0} seconds to decode {1:.2f} seconds of audio.\n'.format(time.time() - decode_start_time, len(audio)/1000.0))

  # Crop to relevant portion of the audio
  if AUDIO_SAMPLE_END_S <= AUDIO_SAMPLE_START_S:
    raise Exception('The audio end time {0} should not come before or equal to {1}'.format(AUDIO_SAMPLE_END_S, AUDIO_SAMPLE_END_S))
  if len(audio) < AUDIO_SAMPLE_END_S*1000:
    raise Exception('The input does not contain {0} seconds of audio. Please use a bigger input or more accurate audio sampling points'.format(AUDIO_SAMPLE_END_S))
  audio = audio[AUDIO_SAMPLE_START_S*1000:AUDIO_SAMPLE_END_S*1000]

  # np.int32 or np.int16? Opening in Audacity seems to reveal the mp3 file as
  # 32-bit float
  data = np.fromstring(audio._data, np.int32)

  # What is the correct way to separate data into channels?
  # channels = []
  # for channel in xrange(audio.channels):
  #   channels.append(data[channel::audio.channels])
  return path, data, STANDARD_FREQUENCY

def  convert_mp3_to_standard_form(path, id3tags):
  """
  Converts mp3 file to the standard format of a Constant Bitrate,
  STANDARD_BITRATE kbps and Sample rate STANDARD_FREQUENCY Hz if it is
  not already in that format
  """
  os.rename(path, path.decode('utf8').encode('ascii', 'ignore'))
  path = path.decode('utf8').encode('ascii', 'ignore')
  if id3tags.info.bit_rate_str == '{0} kb/s'.format(STANDARD_BITRATE) and \
      id3tags.info.sample_freq == STANDARD_FREQUENCY:
    return path
  try:
    sbp.check_output(['which', 'ffmpeg'])
  except sbp.CalledProcessError:
    raise Exception('Could not locate ffmpeg on your system.')
  print('Converting {0} to the standard mp3 format...'.format(path))
  converstion_start_time = time.time()
  try:
    filepath, _ = os.path.splitext(path)
    tmp_filepath = '{0}1.mp3'.format(filepath)
    sbp.check_call(['ffmpeg', '-i', path, '-codec:a','libmp3lame','-b:a','{0}k'.format(STANDARD_BITRATE),'-ar','{0}'.format(STANDARD_FREQUENCY),'-loglevel','quiet','-y', tmp_filepath])
    os.remove(path)
    os.rename(tmp_filepath, path)
  except:
    raise Exception('Something went wrong when running ffmpeg.')
  print('Done converting file in {0} seconds.\n'.format(time.time() - converstion_start_time))
  return path

def print_song_metadata(id3tags):
  print(u'---------- SONG DETAILS ----------')
  print(u'Title:            {0}'.format(id3tags.tag.title))
  print(u'Artist:           {0}'.format(id3tags.tag.artist))
  print(u'Album:            {0}'.format(id3tags.tag.album))

def print_audio_metadata(id3tags):
  print(u'---------- AUDIO DETAILS ----------')
  print(u'Bitrate:          {0}'.format(id3tags.info.bit_rate_str))
  print(u'Mode:             {0}'.format(id3tags.info.mode))
  print(u'Sample Frequency: {0}'.format(id3tags.info.sample_freq))
  print(u'Size (in bytes):  {0}'.format(id3tags.info.size_bytes))
  print(u'Time (in s):      {0}'.format(id3tags.info.time_secs))

