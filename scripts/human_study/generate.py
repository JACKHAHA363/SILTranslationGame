import argparse
import os
import random
import csv
import pickle

parser = argparse.ArgumentParser()
parser.add_argument('-txt_dir', required=True)
parser.add_argument('-fr', required=True)
parser.add_argument('-outdir', required=True)
parser.add_argument('-nb_examples', default=25)
parser.add_argument('-nb_csv', default=20)
args = parser.parse_args()

if not os.path.exists(args.outdir):
	os.makedirs(args.outdir)


# Read Eng
files = sorted(os.listdir(args.txt_dir))
hyps = {name: [line.rstrip('\n').replace('&apos;', "'").replace('&quot;', '"')
			   for line in open(os.path.join(args.txt_dir, name))] for name in files}
for name in files:
	print(name, 'has {} line'.format(len(hyps[name])))
	for line in hyps[name]:
		assert "&" not in line, line

# Read French
frs = [line.rstrip('\n').replace('&apos;', "'").replace('&quot;', '"') for line in open(args.fr)]
for line in frs:
	assert '&' not in line, line


ex_ids = [i for i in range(len(frs))]
datas = []
for person_id in range(args.nb_csv):
	data = {'w_src': [], 'wo_src': []}

	# Pick unique ex id for both experiments
	random.shuffle(ex_ids)
	exp2_ex_ids = ex_ids[-args.nb_examples:].copy()

	# Exp1
	with open(os.path.join(args.outdir, 'w_src_{}.csv'.format(person_id)), 'w', newline='') as csvfile:
		writer = csv.writer(csvfile, quoting=csv.QUOTE_NONNUMERIC)
		for test_id, ex_id in enumerate(ex_ids[:args.nb_examples]):
			fr = frs[ex_id]
			writer.writerow(["", fr])
			random.shuffle(files)
			methodA, methodB = files[:2]
			writer.writerow(["A", hyps[methodA][ex_id]])
			writer.writerow(["B", hyps[methodB][ex_id]])
			writer.writerow(["Which is Better: ('A', 'B', or '-')", ""])
			writer.writerow(["", ""])
			data['w_src'].append((ex_id, methodA, methodB))

	# Exp2
	with open(os.path.join(args.outdir, 'wo_src_{}.csv'.format(person_id)), 'w', newline='') as csvfile:
		writer = csv.writer(csvfile, quoting=csv.QUOTE_NONNUMERIC)
		for test_id, ex_id in enumerate(ex_ids[-args.nb_examples:]):
			random.shuffle(files)
			methodA, methodB = files[:2]
			writer.writerow(["A", hyps[methodA][ex_id]])
			writer.writerow(["B", hyps[methodB][ex_id]])
			writer.writerow(["Which is Better: ('A', 'B', or '-')", ""])
			writer.writerow(["", ""])
			data['wo_src'].append((ex_id, methodA, methodB))

	datas.append(data)

with open(os.path.join(args.outdir, 'data.pkl'), 'wb') as f:
	pickle.dump(datas, f)
