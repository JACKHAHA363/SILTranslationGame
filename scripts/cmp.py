import argparse
import os

def get_args():
	parser = argparse.ArgumentParser('')
	parser.add_argument('txts')
	parser.add_argument('out')
	return parser.parse_args()

args = get_args()
files = os.listdir(args.txts)
liness = []
for file in files:
	with open(os.path.join(args.txts, file)) as f:
		lines = [line.rstrip('\n') for line in f.readlines()]
	liness.append(lines)

longest_name = max([len(file) for file in files])
template = "{:<" + str(longest_name) + "}: {}"


nb_sent = len(liness[0])
outputs = []
for i in range(nb_sent):
	out = []
	for file_name, lines in zip(files, liness):
		out.append(template.format(file_name, lines[i]))
	out.append('\n')
	out = '\n'.join(out)
	outputs.append(out)
outputs = '\n'.join(outputs)

with open(args.out, 'w') as f:
	f.write(outputs)
