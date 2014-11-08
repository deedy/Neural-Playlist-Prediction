import numpy as np
import cPickle as pickle
import sys
import os
import time

S3_PATH = os.path.join('/mnt', 's3')
NUM_BATCHES = 15
NUM_SONGS_PER_GENRE = 90
SONG_NUM_OFFSET = 10

def make_batches(in_path, out_path, NUM_NODES, node=None):
	"""
	Make data_batch_* files for cuda-convnet from
	the GTZAN genre data set

	:param str in_path: Path to .pik files containing
	song spectrograms for the GTZAN genre data set
	(e.g. ".../processed")
	:param str out_path: Path to output files
	data_batch_* and batches.meta
	"""

	assert NUM_NODES > 0, "Invalid # of nodes!"

	genres = os.listdir(in_path)	
	#NUM_FEATURES = pickle.load(open(os.path.join(path, genres[0], genres[0] + '.00001.pik'), 'rb')).data.shape[0]
	NUM_FEATURES = 660000

	batch_size = NUM_SONGS_PER_GENRE/NUM_BATCHES*len(genres)
	
	# make batches.meta
	meta = {'num_vis' : NUM_FEATURES, 'data_mean' : np.zeros((NUM_FEATURES,1)), 'num_cases_per_batch' : NUM_SONGS_PER_GENRE/NUM_BATCHES*len(genres), 'label_names' : genres}
	pickle.dump(meta, open(os.path.join(out_path, 'batches.meta'), 'wb'))
	print "Made batches.meta"	

	if node is None and NUM_NODES > 1:
		return

	bx = NUM_BATCHES/NUM_NODES*node
	by = bx + NUM_BATCHES/NUM_NODES
	for b in xrange(bx, by):
		t = time.time()
		batch = {'filenames' : [''] * batch_size, 'batch_label' : 'training ' + str(b) + ' of ' + str(NUM_BATCHES), 'labels' : [-1] * batch_size, 'data' : np.zeros((NUM_FEATURES,batch_size))}
		x = NUM_SONGS_PER_GENRE/NUM_BATCHES*b+SONG_NUM_OFFSET
		y = NUM_SONGS_PER_GENRE/NUM_BATCHES*(b+1)+SONG_NUM_OFFSET
		for g, genre in enumerate(genres):
			for i in xrange(x, y):
				filename = genre + '.' + ('000' if i < NUM_SONGS_PER_GENRE else '00') + str(i) + '.pik'
				batch_idx = NUM_SONGS_PER_GENRE/NUM_BATCHES*g + i - NUM_SONGS_PER_GENRE/NUM_BATCHES*b - SONG_NUM_OFFSET
				#try:
				batch['data'][:,batch_idx] = pickle.load(open(os.path.join(in_path, genre, filename), 'rb')).data[:NUM_FEATURES]
				#except:
					#print "Invalid number of features in " + filename
				batch['labels'][batch_idx] = g
				batch['filenames'][batch_idx] = filename
		pickle.dump(batch, open(os.path.join(out_path, 'data_batch_' + str(b+1)), 'wb'))
		print 'Made batch ' + str(b+1) + " of " + str(NUM_BATCHES) + " in " + str(time.time() - t) + " seconds" 

def main():
	in_path = sys.argv[1]
	out_path = sys.argv[2]
	num_nodes = sys.argv[3].split('--num-nodes=')[1]
	node = sys.argv[4].split('--node=')[1] if len(sys.argv) >= 5 else None
	make_batches(in_path, out_path, num_nodes, node)

if __name__ == '__main__':
	main()
