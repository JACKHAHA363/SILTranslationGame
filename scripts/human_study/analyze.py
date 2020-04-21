"""
Compute pair-wise score
"""
import argparse
import csv

import os
import pickle
import ipdb
import numpy as np
from pandas import DataFrame

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
    win_mat = np.zeros([len(methods), len(methods)])
    count_mat = np.zeros([len(methods), len(methods)])
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
        if score1 > score2:
            win_mat[method1, method2] += 1
        if score2 > score1:
            win_mat[method2, method1] += 1
        count_mat[method1, method2] += 1
        count_mat[method2, method1] += 1
    return win_mat, count_mat


def accumulate_csvs(csv_files):
    total_win_mat = np.zeros([len(methods), len(methods)])
    total_count_mat = np.zeros([len(methods), len(methods)])
    for file in csv_files:
        win_mat, count_mat = read_csv(os.path.join(args.csv_dir, file))
        total_count_mat += count_mat
        total_win_mat += win_mat
        if count_mat.sum() == 0:
            print('0 result', file)
    return total_win_mat, total_count_mat


def visualize(total_win_mat, total_count_mat):
    print('Win Ratio')
    win_ratio = total_win_mat / total_count_mat
    for i in range(len(methods)):
        win_ratio[i, i] = 0
    df = DataFrame(win_ratio, index=methods, columns=methods)
    print(df)

    print('Tie Ratio')
    tie_ratio = np.zeros([len(methods), len(methods)])
    for i in range(len(methods)):
        for j in range(len(methods)):
            tie_ratio[i, j] = 1 - (win_ratio[i, j] + win_ratio[j, i])
    df = DataFrame(tie_ratio,
                   index=methods, columns=methods)
    print(df)

    print('Adjusted Win Ratio')
    adjusted_win_ratio = np.zeros([len(methods), len(methods)])
    for i in range(len(methods)):
        for j in range(len(methods)):
            if i == j:
                continue
            adjusted_win_ratio[i, j] = win_ratio[i, j] / float(win_ratio[i, j] + win_ratio[j, i])
    print(DataFrame(np.concatenate([adjusted_win_ratio, adjusted_win_ratio.sum(-1).reshape([-1, 1])], -1),
                    index=methods, columns=methods + ['SUM']))

    print('Nb. pairs tested:', total_count_mat.sum())
    df = DataFrame(total_count_mat,
                   index=methods, columns=methods)
    print(df)

wo_total_win_mat, wo_total_count_mat = accumulate_csvs(wo_src)
w_total_win_mat, w_total_count_mat = accumulate_csvs(w_src)
print('############# Without French #############')
visualize(wo_total_win_mat, wo_total_count_mat)

print('############## With French ###############')
visualize(w_total_win_mat, w_total_count_mat)
