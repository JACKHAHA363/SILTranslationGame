import argparse
import os
import random
import csv
import string
from itertools import product
import pickle
import datetime


def randomString(stringLength=10):
	""" Generate a random string of fixed length """
	letters = string.ascii_lowercase + string.ascii_uppercase + string.digits
	return ''.join(random.choice(letters) for i in range(stringLength))


parser = argparse.ArgumentParser()
parser.add_argument('-data', default=None)
parser.add_argument('-txt_dir', default=None)
parser.add_argument('-fr', default=None)
parser.add_argument('-outdir', required=True)
parser.add_argument('-nb_examples', default=25, type=int)
parser.add_argument('-nb_csv', default=20, type=int)
args = parser.parse_args()

if not os.path.exists(args.outdir):
	os.makedirs(args.outdir)
csv_dir = os.path.join(args.outdir, datetime.datetime.now().strftime("%m%d%H_%M_%S") +
					   "people{}_ex{}".format(args.nb_csv, args.nb_examples))
if not os.path.exists(csv_dir):
	os.makedirs(csv_dir)

if args.data is not None:
	print('Load data from', args.data)
	data = pickle.load(open(args.data, 'rb'))
	frs = data['frs']
	hyps = data['hyps']
	info2id = data['info2id']
	id2info = data['id2info']
	methods = data['methods']
else:
	print('Read text from', args.txt_dir)
	assert args.txt_dir is not None and args.fr is not None

	# Read Eng
	methods = sorted(os.listdir(args.txt_dir))
	hyps = {name: [line.rstrip('\n').replace('&apos;', "'").replace('&quot;', '"')
				   for line in open(os.path.join(args.txt_dir, name))] for name in methods}
	for name in methods:
		print(name, 'has {} line'.format(len(hyps[name])))
		for line in hyps[name]:
			assert "&" not in line, line

	# Read French
	frs = [line.rstrip('\n').replace('&apos;', "'").replace('&quot;', '"') for line in open(args.fr)]
	for line in frs:
		assert '&' not in line, line

	while True:
		sent_ids = [randomString() for _ in range(len(methods) * len(frs))]
		if len(sent_ids) == len(set(sent_ids)):
			print('Sentence ids generated!')
		break

	info2id, id2info = {}, {}
	for idx, info in enumerate(product(methods, range(len(frs)))):
		sent_id = sent_ids[idx]
		info2id[info] = sent_id
		id2info[sent_id] = info

	pickle.dump({'info2id': info2id,
				 'id2info': id2info,
				 'hyps': hyps,
				 'frs': frs,
				 'methods': methods}, open(os.path.join(args.outdir, 'data.pkl'), 'wb'))


ex_ids = [i for i in range(len(frs))]
for person_id in range(args.nb_csv):
	# Pick unique ex id for both experiments
	random.shuffle(ex_ids)
	exp2_ex_ids = ex_ids[-args.nb_examples:].copy()

	# Exp1
	with open(os.path.join(csv_dir, 'w_src_{}.csv'.format(person_id)), 'w', newline='') as csvfile:
		writer = csv.writer(csvfile, quoting=csv.QUOTE_NONNUMERIC)
		for test_id, ex_id in enumerate(ex_ids[:args.nb_examples]):
			fr = frs[ex_id]
			writer.writerow(["", "", fr])
			random.shuffle(methods)
			methodA, methodB = methods[:2]
			writer.writerow([info2id[(methodA, ex_id)], "", hyps[methodA][ex_id]])
			writer.writerow([info2id[(methodB, ex_id)], "", hyps[methodB][ex_id]])
			writer.writerow(["", "", ""])

	# Exp2
	with open(os.path.join(csv_dir, 'wo_src_{}.csv'.format(person_id)), 'w', newline='') as csvfile:
		writer = csv.writer(csvfile, quoting=csv.QUOTE_NONNUMERIC)
		for test_id, ex_id in enumerate(ex_ids[-args.nb_examples:]):
			random.shuffle(methods)
			methodA, methodB = methods[:2]
			writer.writerow([info2id[(methodA, ex_id)], "", hyps[methodA][ex_id]])
			writer.writerow([info2id[(methodB, ex_id)], "", hyps[methodB][ex_id]])
			writer.writerow(["", "", ""])
