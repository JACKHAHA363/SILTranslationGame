import argparse
import os
from collections import Counter

def get_args():
	parser = argparse.ArgumentParser('')
	parser.add_argument('txts')
	return parser.parse_args()


def get_counter(file_path):
	c = Counter()
	with open(file_path) as f:
		for line in f:
			words = line.rstrip('\n').split()
			for word in words:
				c[word] += 1
	return c


args = get_args()
files = os.listdir(args.txts)
counters = {}
for file in files:
	counters[file] = get_counter(os.path.join(args.txts, file))

for file, counter in counters.items():

import ipdb; ipdb.set_trace()

