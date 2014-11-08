import sys
import os

def main():
  in_path = sys.argv[1]
  out_path = sys.argv[2]
  nodes = int(sys.argv[3].split('--num-nodes=')[1])
  cmd = 'cd Neural-Playlist-Prediction; python data.py ' + in_path + ' ' + out_path + ' --num-nodes=' + str(nodes)
  for i in xrange(nodes):
    os.system('ssh node' + ('00' if i < 10 else '0') + str(i+1) + ' \'' + cmd + ' --node=' + str(i)  + '\' &')
    #os.system(cmd + ' --node=' + str(i))

if __name__ == '__main__':
  main()
