import argparse
import os
import nltk
from collections import Counter
import matplotlib.pyplot as plt
from tqdm import tqdm
from nltk import word_tokenize


def get_args():
	parser = argparse.ArgumentParser('')
	parser.add_argument('txts')
	parser.add_argument('-min_freq', type=int, default=50)
	return parser.parse_args()


args = get_args()
files = os.listdir(args.txts)
files = ['ref', 'NIL', 'S2P1', 'S2P5', 'Gumbel', 'Pretrain']
ref_id = files.index('ref')
liness = []
for file in files:
	with open(os.path.join(args.txts, file)) as f:
		lines = [line.rstrip('\n') for line in f.readlines()]
	liness.append(lines)

longest_name = max([len(file) for file in files])
template = "{:<" + str(longest_name) + "}: {}"


# Get all post info
pos_dict = {}
for file in tqdm(files):
	tags = []
	file_id = files.index(file)
	for lines in zip(*liness):
		sent = lines[file_id]
		tags.append(nltk.pos_tag(word_tokenize(sent)))
	pos_dict[file] = tags

# Get frequency of each tag
EXCLUDES = ['0.00%']
total = Counter()
for ref_tags in pos_dict['ref']:
	for _, ref_tag in ref_tags:
		total[ref_tag] += 1
TAGS = [tag for tag in total if total[tag] >= args.min_freq and tag != EXCLUDES]
TAGS = sorted(TAGS, key=lambda t:total[t], reverse=True)
print(TAGS)

# POS dist
pos_dist_dict = {}
for file in tqdm(files):
	total_tags = 0
	pos_dist = Counter()
	for tags in pos_dict[file]:
		for word, tag in tags:
			pos_dist[tag] += 1
			total_tags += 1
	for tag in TAGS:
		pos_dist[tag] /= float(total_tags)
	pos_dist_dict[file] = pos_dist
fig, ax = plt.subplots()
for file, pos_dist in pos_dist_dict.items():
	dist = [pos_dist[tag] for tag in TAGS]
	line, = ax.plot(dist)
	line.set_label(file)
ax.legend()
fig.savefig('pos_dist.png')


from pandas import DataFrame
result = DataFrame(columns=TAGS)
for file in tqdm(files):
	if file == 'ref':
		continue
	match = Counter()
	for ref_tags, hyp_tags in zip(pos_dict['ref'], pos_dict[file]):
		hyp_words = set([word for word, tag in hyp_tags])
		for word, ref_tag in ref_tags:
			if word in hyp_words:
				match[ref_tag] += 1

	# Record
	for tag in TAGS:
		result.loc[file, tag] = match[tag] / float(total[tag])
result.to_csv('recall.csv')
print(result)