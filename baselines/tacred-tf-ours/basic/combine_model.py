import sys,os
import argparse
import numpy as np
from collections import Counter
import scorer
import data_utils

NUM_CLASS = 42

def read_prob_file(filename):
    all_probs_list = []
    with open(filename, 'r') as infile:
        for line in infile:
            array = line.strip().split()
            if len(array) != NUM_CLASS + 1:
                continue
            probs = [float(x) for x in array[1:]]
            assert len(probs) == NUM_CLASS
            all_probs_list.append(probs)
    return np.array(all_probs_list)

def majority_vote(prob_matrices):
    if not isinstance(prob_matrices, list):
        raise Exception('Need a prob matrix list for doing majority_vote.')
    all_preds = []
    num_examples, num_class = prob_matrices[0].shape
    for pm in prob_matrices:
        pred_idx = np.argmax(pm, axis=1)
        all_preds.append(pred_idx.tolist())
    preds = []
    for i in range(num_examples):
        c = Counter()
        for p in all_preds:
            c[p[i]] += 1
        preds.append(c.most_common()[0][0])
    return preds

def main():
    parser = argparse.ArgumentParser(description='Take the probability files from several models and do model ensembling.')
    parser.add_argument('--prob_dir', type=str, default='tmp/ensemble/', dest='prob_dir', action='store', help='The dir where the prob files locate.')
    parser.add_argument('--files', type=str, dest='files', action='store', required=True, help='The list of filenames, separated by comma.')
    parser.add_argument('--key_file', type=str, dest='key_file', action='store', required=True, help='Where to find the key file.')

    args = parser.parse_args()

    if not os.path.exists(args.prob_dir):
        raise Exception('Probability file dir does not exist at ' + args.prob_dir)
    prob_files = args.files.split(',')
    if len(prob_files) <= 1:
        raise Exception('Need to provide more than one model prediction files.')
    prob_files = [os.path.join(args.prob_dir, x) for x in prob_files]

    prob_matrices = []
    for fname in prob_files:
        print "Reading prediction prob file at: " + fname
        pm = read_prob_file(fname)
        prob_matrices.append(pm)

    print "Doing majority vote to generate final predictions..."
    preds = majority_vote(prob_matrices)
    
    # convert preds from index to labels
    label2id = data_utils.LABEL_TO_ID
    id2label = dict([(v,k) for k,v in label2id.items()])
    preds = [id2label[x] for x in preds]
    
    # write pred file
    pred_file = args.prob_dir + '/ensemble.prediction.tmp'
    with open(pred_file, 'w') as outfile:
        for p in preds:
            print >> outfile, p + '\t1.0'
    
    # score
    scorer.score(args.key_file, [pred_file], 1, True)

if __name__ == '__main__':
    main()
