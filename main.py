from optparse import OptionParser
from parse_mp3 import parse_mp3

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
  format = path[path.rindex('.')+1:]

  parsers = {
    'mp3': parse_mp3
  }

  if not format in parsers:
    raise Exception('Do not know how to decode {0}'.format(path))
  parsers[format](path)






