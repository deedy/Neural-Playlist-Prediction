import sys
import os

def main():
	nodes=int(sys.argv[1].split('--num-nodes=')[1])
	for i in xrange(nodes):
		r = lambda x: 'ssh node' + ('00' if i < 10 else '0') + str(i+1) + ' \''+x+'\''
		# s3fs
		os.system(r('mkdir /mnt/s3; s3fs 4780 /mnt/s3'))
		# copy Neural-Playlist-Prediction from s3
		os.system(r('cp -r /mnt/s3/Neural-Playlist-Prediction .'))

if __name__ == '__main__':
	main()
