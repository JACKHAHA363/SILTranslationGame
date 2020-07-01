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
from scipy.stats import binom_test

parser = argparse.ArgumentParser()
parser.add_argument('-csv_dir', required=True)
parser.add_argument('-data', required=True)
args = parser.parse_args()

with open(args.data, 'rb') as f:
    data = pickle.load(open(args.data, 'rb'))
methods = data['methods']
id2info = data['id2info']
hyps = data['hyps']
csv_files = [f for f in os.listdir(args.csv_dir) if '.csv' in f]
wo_src = [csv_file for csv_file in csv_files if 'wo_src' in csv_file]
w_src = [csv_file for csv_file in csv_files if 'w_src' in csv_file]


def retrieval_method(sent_id, sent):
    """ Return the method ID """
    try:
        return methods.index(id2info[sent_id][0])
    except KeyError:
        pass

    # Try to find by bruteforcing
    found = []
    for method in hyps:
        for target in hyps[method]:
            if sent == target:
                found.append(method)
    found = list(set(found))
    if len(found) == 1:
        return methods.index(found[0])
    else:
        return None


def read_csv(csv_path):
    """ Read info from csv. Throw the exception if CSV is invalid """
    state = 'init'
    results = []
    buffer = []
    nb_exp = 0
    with open(csv_path) as csv_file:
        reader = csv.reader(csv_file, delimiter=',')
        for line in reader:
            if len(line) != 3:
                ipdb.set_trace()
            sent_id, score, sent = line
            if state == 'init':
                if sent_id == "":
                    continue
                else:
                    method = retrieval_method(sent_id, sent)
                    if method is not None:
                        buffer.append((method, score))
                    state = "one"
            elif state == "one":
                if sent_id == "":
                    assert False, "Something is wrong at this row {} {} {}".format(sent_id, score, sent)
                else:
                    method = retrieval_method(sent_id, sent)
                    if method is not None:
                       buffer.append((method, score))
                    if len(buffer) == 2:
                        results.append(buffer)
                    buffer = []
                    state = 'init'
                    nb_exp += 1
    print('{}:\t{}\t{}%:'.format(csv_path, len(results),
                               len(results) / float(nb_exp) *100))
    assert len(results) > 0

    # Deal with results
    win_mat = np.zeros([len(methods), len(methods)])
    count_mat = np.zeros([len(methods), len(methods)])
    for result in results:
        method1, score1 = result[0]
        method2, score2 = result[1]
        if score1 == "" and score2 == "":
            continue
        score1 = 0 if score1 == "" else int(score1)
        score2 = 0 if score2 == "" else int(score2)
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
    #print('Win Ratio')
    win_ratio = total_win_mat / total_count_mat
    #for i in range(len(methods)):
    #    win_ratio[i, i] = 0
    #df = DataFrame(win_ratio, index=methods, columns=methods)
    #print(df)

    #print('Tie Ratio')
    #tie_ratio = np.zeros([len(methods), len(methods)])
    #for i in range(len(methods)):
    #    for j in range(len(methods)):
    #        tie_ratio[i, j] = 1 - (win_ratio[i, j] + win_ratio[j, i])
    #df = DataFrame(tie_ratio,
    #               index=methods, columns=methods)
    #print(df)

    print('Adjusted Win Ratio')
    adjusted_win_ratio = np.zeros([len(methods), len(methods)])
    for i in range(len(methods)):
        for j in range(len(methods)):
            if i == j:
                continue
            adjusted_win_ratio[i, j] = win_ratio[i, j] / float(win_ratio[i, j] + win_ratio[j, i])
    print(DataFrame(np.concatenate([adjusted_win_ratio, adjusted_win_ratio.sum(-1).reshape([-1, 1])], -1),
                    index=methods, columns=methods + ['SUM']))

    print('Adjusted Count')
    adjusted_count = np.zeros([len(methods), len(methods)])
    for i in range(len(methods)):
        for j in range(len(methods)):
            if i == j:
                continue
            adjusted_count[i, j] = total_count_mat[i, j] * float(win_ratio[i, j] + win_ratio[j, i])
    print(DataFrame(adjusted_count,
                    index=methods,
                    columns=methods))
    print('Nb. pairs tested:', adjusted_count.sum() / 2)

    print('P-values for single sided binomial test')
    single_sided_p_value = np.zeros([len(methods), len(methods)])
    for i in range(len(methods)):
        for j in range(len(methods)):
            if i == j:
                continue
            n = adjusted_count[i,j]
            x_win_ratio = adjusted_win_ratio[i,j] if adjusted_win_ratio[i,j] > 0.5 else adjusted_win_ratio[j,i]
            x = int(n * x_win_ratio)
            single_sided_p_value[i,j] = binom_test(x, n)
    print(DataFrame(single_sided_p_value,
                    index=methods,
                    columns=methods))

wo_total_win_mat, wo_total_count_mat = accumulate_csvs(wo_src)
w_total_win_mat, w_total_count_mat = accumulate_csvs(w_src)
print('############# Without French #############')
visualize(wo_total_win_mat, wo_total_count_mat)

print('############## With French ###############')
visualize(w_total_win_mat, w_total_count_mat)
