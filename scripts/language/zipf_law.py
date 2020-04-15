import argparse
import os
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np


def get_args():
	parser = argparse.ArgumentParser('')
	parser.add_argument('txts')
	parser.add_argument('-min_freq', default=1, type=int, help='min freq to discard data')
	return parser.parse_args()


def get_counter(file_path, min_freq):
	c = Counter()
	with open(file_path) as f:
		for line in f:
			words = line.rstrip('\n').split()
			for word in words:
				c[word] += 1
	return {word: freq for word, freq in c.items() if freq >= min_freq}


args = get_args()
files = os.listdir(args.txts)
counters = {}
for file in files:
	counters[file] = get_counter(os.path.join(args.txts, file), args.min_freq)

# Plot Token Freq
fig, ax = plt.subplots(figsize=(8, 6))
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
for file in files:
	counter = counters[file]
	sorted_freqs = np.array(sorted(counter.values(), reverse=True))
	ranks = np.array([i+1 for i in range(len(sorted_freqs))])
	line, = ax.plot(np.log(ranks), np.log(sorted_freqs))
	line.set_label(file)
ax.legend(fontsize=20)
ax.set_xlabel('log Rank', fontsize=20)
ax.set_ylabel('Log Freq', fontsize=20)
fig.savefig('freqs.png')

# Plot Token Freq diff
sorted_ref_freqs = np.array(sorted(counters['ref'].values(), reverse=True))
fig, ax = plt.subplots(figsize=(8, 6))
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
for file in files:
	counter = counters[file]
	sorted_freqs = np.array(sorted(counter.values(), reverse=True))
	final_len = min(len(sorted_freqs), len(sorted_ref_freqs))
	ranks = np.array([i+1 for i in range(len(sorted_freqs))])
	line, = ax.plot(np.log(ranks)[:final_len], np.log(sorted_freqs)[:final_len] - np.log(sorted_ref_freqs)[:final_len])
	line.set_label(file)
ax.legend(fontsize=20)
ax.set_xlabel('log Rank', fontsize=20)
ax.set_ylabel('Log Freq Diff', fontsize=20)
fig.savefig('freqs_diff.png')