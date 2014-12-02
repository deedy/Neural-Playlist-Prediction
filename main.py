from optparse import OptionParser
from parse_mp3 import parse_mp3
from parse_gtzan import parse_gtzan
from AudioBite import AudioBite
import os
import glob

from IPython.core.debugger import Tracer

SAVE_DIR = 'data'
PARSE_METHODS = {'.mp3': parse_mp3, '.au': parse_gtzan}

def run(path):
  to_process = []
  if os.path.isfile(path):
    fp, ext = os.path.splitext(path)
    if not ext in PARSE_METHODS:
      raise Exception('Sorry, we currently only support the following types: {0}'.format('\n'.join(PARSE_METHODS.keys())))
    to_process = [ path ]
  elif os.path.isdir(path):
    audio_files = [os.path.join(path, audio_file) for audio_file in os.listdir(path) if os.path.isfile(os.path.join(path, audio_file)) and os.path.splitext(audio_file)[1] in PARSE_METHODS]

    if len(audio_files) == 0:
      raise Exception('Sorry, we currently only support the following file types: {0}, and none of those types were found in {1}'.format(' '.join(PARSE_METHODS.keys()), path))
    to_process = audio_files

    # Add option to reprocess all
    unprocessed_files = []
    for audio_file in audio_files:
      filepath, ext  = os.path.splitext(audio_file)
      savepath = os.sep.join([SAVE_DIR] + filepath.split(os.sep)[1:])
      processed_files = glob.glob(savepath+'*')
      if len(processed_files) == 0:
        unprocessed_files.append(audio_file)
    to_process = unprocessed_files
    print('Beginning to parse {0} mp3 files in {1}'.format(len(to_process), path))
  else:
    raise Exception('{0} is not a valid directory or path'.format(path))

  for audio_file in to_process:
    _, ext  = os.path.splitext(audio_file)
    path, data, freq = PARSE_METHODS[ext](audio_file)
    processed_audio = AudioBite(path, data, freq)
    # This is where the fun stuff happens
    processed_audio.save_spectrogram()
    processed_audio.save_mel_spectrogram()
    processed_audio.save_mfcc()
    processed_audio.save()

if __name__ == '__main__':
  usage = "usage: %prog [options] arg"
  parser = OptionParser(usage)
  (options, args) = parser.parse_args()

  if len(args) == 0:
    raise Exception('No arguments supplied. Failing.')
    sys.exit
  if len(args) > 1:
    print('Only accepts one argument at this moment. Using first argument {0}'.format(args[0]))
  path = args[0]
  main(path)