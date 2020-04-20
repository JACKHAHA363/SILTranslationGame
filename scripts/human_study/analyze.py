"""
Compute pair-wise score
"""
import argparse
import csv

import os
import pickle
import ipdb
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('-csv_dir', required=True)
parser.add_argument('-idmap', required=True)
args = parser.parse_args()

with open(args.idmap, 'rb') as f:
    idmap = pickle.load(open(args.idmap, 'rb'))
methods = idmap['methods']
id2info = idmap['id2info']
csv_files = [f for f in os.listdir(args.csv_dir) if '.csv' in f]
wo_src = [csv_file for csv_file in csv_files if 'wo_src' in csv_file]
w_src = [csv_file for csv_file in csv_files if 'w_src' in csv_file]


def read_csv(csv_path):
    state = 'init'
    results = []
    buffer = []
    with open(csv_path) as csv_file:
        reader = csv.reader(csv_file, delimiter=',')
        try:
            for line in reader:
                if len(line) != 3:
                    ipdb.set_trace()
                sent_id, score, sent = line
                if state == 'init':
                    if sent_id == "":
                        continue
                    else:
                        buffer.append((sent_id, score))
                        state = "one"
                elif state == "one":
                    if sent_id == "":
                        assert False, "Something is wrong at this row {} {} {}".format(sent_id, score, sent)
                    else:
                        buffer.append((sent_id, score))
                        results.append(buffer)
                        buffer = []
                        state = 'init'
        except ValueError:
            ipdb.set_trace()

    # Deal with results
    conf_mat = np.zeros([len(methods), len(methods)])
    nb_tests = 0
    for result in results:
        id1, score1 = result[0]
        id2, score2 = result[1]
        try:
            method1 = methods.index(id2info[id1][0])
            method2 = methods.index(id2info[id2][0])
        except KeyError:
            if id1 not in id2info:
                print('{} not found'.format(id1))
            if id2 not in id2info:
                print('{} not found'.format(id2))
            continue
        if score1 == "" or score2 == "":
            continue
        score1 = int(score1)
        score2 = int(score2)
        nb_tests += 1
        if score1 > score2:
            conf_mat[method1, method2] += 1
            conf_mat[method2, method1] -= 1
        elif score1 < score2:
            conf_mat[method1, method2] -= 1
            conf_mat[method2, method1] += 1
    return conf_mat, nb_tests


w_src_tests = 0
w_src_conf_mat = np.zeros([len(methods), len(methods)])
for w_src_file in w_src:
    conf_mat, nb_tests = read_csv(os.path.join(args.csv_dir, w_src_file))
    if nb_tests == 0:
        print('0 result', w_src_file)
    w_src_tests += nb_tests
    w_src_conf_mat += conf_mat

wo_src_tests = 0
wo_src_conf_mat = np.zeros([len(methods), len(methods)])
for wo_src_file in wo_src:
    conf_mat, nb_tests = read_csv(os.path.join(args.csv_dir, wo_src_file))
    wo_src_tests += nb_tests
    if nb_tests == 0:
        print('0 result', wo_src_file)
    wo_src_conf_mat += conf_mat


def visualize(conf_mat, nb_tests):
    from pandas import DataFrame
    df = DataFrame(np.concatenate([conf_mat, conf_mat.sum(-1).reshape([-1, 1])], -1),
                   index=methods, columns=methods + ['SUM'])
    print(df)
    print('Nb. pairs tested:', nb_tests)


print('With French')
visualize(w_src_conf_mat, w_src_tests)
print('Without French')
visualize(wo_src_conf_mat, wo_src_tests)

def rank_by_winning_score(conf_mat):
    scores = conf_mat.sum(-1)
    print('')

