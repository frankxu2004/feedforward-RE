#!/usr/bin/python

"""
This is used to get Vocab for all fields in the dataset.
"""

from stanza.text.dataset import Dataset
from collections import Counter
from collections import OrderedDict
import argparse

SKIP_TOKEN = '-'
PAD_TOKEN = '<PAD>'
UNK_TOKEN = '<UNK>'


def get_counter_for_field(filelist, field):
    c = Counter()
    for fin_name in filelist:
        print('loading {}'.format(fin_name))
        d = Dataset.load_conll(fin_name)
        for i, row in enumerate(d):
            for j in range(len(row['word'])):
                t = row[field][j]
                if t == SKIP_TOKEN or t is None:
                    continue
                else:
                    c[t] += 1
    return c


def print_dict(d):
    s = "{"
    for key in d:
        s += "\'%s\': %s, " % (key, d[key])  # Fixed syntax
    s = s[:-2]
    s += "}"
    return s


def main():
    parser = argparse.ArgumentParser(description='Used to get vocab for all fields in TACRED.')
    parser.add_argument('--field', type=str, default='stanford_ner', help='The field to consider.')
    parser.add_argument('--data_dir', type=str, default='../data/tacred', help='The root directory for the data.')

    args = parser.parse_args()
    field = args.field
    data_dir = args.data_dir

    print "Getting vocab for field %s from data in %s ..." % (field, data_dir)
    c = get_counter_for_field([data_dir + '/train.conll', data_dir + '/dev.conll', data_dir + '/test.conll'], field)
    tokens, _ = zip(*c.most_common())
    # add pad token at the beginning
    tokens = [PAD_TOKEN, UNK_TOKEN] + list(tokens)
    token2id = OrderedDict([(t, i) for i, t in enumerate(tokens)])
    print "Total entries: %d" % len(token2id)

    print print_dict(token2id)


if __name__ == '__main__':
    main()
