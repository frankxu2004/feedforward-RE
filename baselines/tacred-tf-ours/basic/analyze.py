import time
import os
import sys
import random
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

import data_utils

tf.app.flags.DEFINE_string('data_dir', '../data/basic/', 'Directory of the data')
tf.app.flags.DEFINE_string('train_dir', './train/', 'Directory to save training checkpoint files')
tf.app.flags.DEFINE_integer('num_class', '42', 'Number of total classes')
tf.app.flags.DEFINE_integer('vocab_size', 35888, 'Vocabulary size')
tf.app.flags.DEFINE_integer('sent_len', 96, 'Input sentence length. This is after the padding is performed.')

FLAGS = tf.app.flags.FLAGS

def get_confusion_matrix():
    key_file = os.path.join(FLAGS.train_dir, 'shuffled.test.key.tmp')
    pred_file = os.path.join(FLAGS.train_dir, 'shuffled.test.prediction.tmp')
    key_labels = read_key_pred_file(key_file)
    pred_labels = read_key_pred_file(pred_file)
    assert len(key_labels) == len(pred_labels)
    label2id = data_utils.load_from_dump(os.path.join(FLAGS.data_dir, 'label2id.dict'))
    id2label = {v:k for k,v in label2id.iteritems()}

    conf_matrix = np.zeros([len(label2id), len(label2id)])
    for k,p in zip(key_labels, pred_labels):
        conf_matrix[label2id[k], label2id[p]] += 1
    print "Constructed confusion matrix with size %d x %d" % tuple(conf_matrix.shape)
    normalized_conf_matrix = conf_matrix / conf_matrix.sum(axis=1)[:,np.newaxis]

    conf_matrix_file = key_file = os.path.join(FLAGS.train_dir, 'conf.matrix.test.tmp')
    normalized_conf_matrix_file = os.path.join(FLAGS.train_dir, 'conf.matrix.normalized.test.tmp')
    write_confusion_matrix(conf_matrix_file, conf_matrix, id2label, force_int=True)
    write_confusion_matrix(normalized_conf_matrix_file, normalized_conf_matrix, id2label)
    print "Confusion matrix written to files."
    plot_confusion_matrix(normalized_conf_matrix, id2label)

def read_key_pred_file(filename):
    labels = []
    with open(filename, 'r') as infile:
        for line in infile:
            array = line.strip().split('\t')
            if len(array) > 2:
                continue # invalid line
            labels.append(array[0])
    return labels

def read_prob_file(filename):
    labels = []
    all_probs_list = []
    with open(filename, 'r') as infile:
        for line in infile:
            array = line.strip().split()
            if len(array) != FLAGS.num_class + 1:
                continue
            labels.append(array[0])
            probs = [float(x) for x in array[1:]]
            assert len(probs) == FLAGS.num_class
            all_probs_list.append(probs)
    return labels, np.array(all_probs_list)

def plot_confusion_matrix(normalized_conf_matrix, id2label):
    fig = plt.figure()
    plt.clf()
    ax = fig.add_subplot(111)
    ax.set_aspect(1)
    res = ax.imshow(np.array(normalized_conf_matrix), cmap=plt.cm.jet, 
                    interpolation='nearest')

    num = normalized_conf_matrix.shape[0]

    # for x in xrange(width):
    #     for y in xrange(height):
    #         ax.annotate(str(conf_arr[x][y]), xy=(y, x), 
    #                     horizontalalignment='center',
    #                     verticalalignment='center')

    cb = fig.colorbar(res)
    label_array = [id2label[i] for i in range(num)]
    plt.xticks(range(num), label_array, rotation=90)
    plt.yticks(range(num), label_array)
    fig.subplots_adjust(bottom=0.3, left=0)
    plt.show()
    return

def write_confusion_matrix(filename, conf_matrix, id2label, force_int=False):
    assert conf_matrix.shape[0] == len(id2label)
    with open(filename, 'w') as outfile:
        # write header line with predicted labels
        outfile.write("#")
        for i in range(len(id2label)):
            outfile.write("\t" + id2label[i])
        outfile.write("\n")
        # write matrix with actual labels
        conf_list = conf_matrix.tolist()
        for i in range(len(id2label)):
            row = conf_list[i]
            if force_int:
                row = [str(int(x)) for x in row]
            else:
                row = [str(x) for x in row]
            outfile.write(id2label[i] + "\t" + '\t'.join(row) + '\n')
    return



def analyze_no_relations():
    key_file = os.path.join(FLAGS.train_dir, 'shuffled.test.key.tmp')
    prob_file = os.path.join(FLAGS.train_dir, 'shuffled.test.probs.tmp')
    key_labels = read_key_pred_file(key_file)
    pred_labels, all_probs = read_prob_file(prob_file)
    label2id = data_utils.load_from_dump(os.path.join(FLAGS.data_dir, 'label2id.dict'))

    correct_probs = [] # the probs of the correct labels
    predicted_probs = []
    prob_diffs = [] # the diff between probs of correct label and predicted label
    for i, correct in enumerate(key_labels):
        pred = pred_labels[i]
        if pred == 'no_relation' and pred != correct: # wrongly predict as no_relation
            correct, pred = label2id[correct], label2id[pred]
            correct_prob = all_probs[i, correct]
            pred_prob = all_probs[i, pred]
            correct_probs.append(correct_prob)
            predicted_probs.append(pred_prob)
            assert pred_prob >= correct_prob
            prob_diffs.append(pred_prob - correct_prob)
    print max(prob_diffs)
    print min(prob_diffs)
    plt.figure()
    plt.subplot(131)
    plt.hist(correct_probs, 50, facecolor='blue', alpha=0.75)
    plt.title("Histogram of P(correct label)\n when there is a relation\n but predict to be no_relation")
    plt.subplot(132)
    plt.hist(predicted_probs, 50, facecolor='red', alpha=0.75)
    plt.title("Histogram of P(predicted label)\n when there is a relation\n but predict to be no_relation")
    plt.subplot(133)
    plt.hist(prob_diffs, 50, facecolor='green', alpha=0.75)
    plt.title("Histogram of P(predicted) - P(correct)\n when there is a relation\n but predict to be no_relation")
    plt.show()

def get_unk_count_in_dataset(loader):
    total_count, unk_count = 0, 0
    padded_sentences = []
    for i in range(loader.batch_size):
        padded_sentences += loader.next_batch()[0][data_utils.WORD_FIELD]
    padded_sentences += loader.get_residual()[0][data_utils.WORD_FIELD]
    for sent in padded_sentences:
        for t in sent:
            if t == data_utils.UNK_ID:
                unk_count += 1
            if t != data_utils.PAD_ID:
                total_count += 1
    return unk_count, total_count

def analyze_unk():
    corruption_prob = 0.04
    print "Loading data using vocab size %d..." % FLAGS.vocab_size
    word2id = data_utils.load_from_dump(FLAGS.data_dir + '%d.vocab' % FLAGS.vocab_size)
    train_loader = data_utils.DataLoader(FLAGS.data_dir + 'train.vocab%d.id' % FLAGS.vocab_size, 50, FLAGS.sent_len, unk_prob=corruption_prob)
    dev_loader = data_utils.DataLoader(FLAGS.data_dir + 'dev.vocab%d.id' % FLAGS.vocab_size, 50, FLAGS.sent_len)
    test_loader = data_utils.DataLoader(FLAGS.data_dir + 'test.vocab%d.id' % FLAGS.vocab_size, 50, FLAGS.sent_len)

    print "Counting..."
    train_unk, train_total = get_unk_count_in_dataset(train_loader)
    dev_unk, dev_total = get_unk_count_in_dataset(dev_loader)
    test_unk, test_total = get_unk_count_in_dataset(test_loader)

    print "Training token count:"
    print "\tunk:%d\ttotal:%d\tratio:%g" % (train_unk, train_total, 1.0*train_unk/train_total)
    print "Dev token count:"
    print "\tunk:%d\ttotal:%d\tratio:%g" % (dev_unk, dev_total, 1.0*dev_unk/dev_total)
    print "Test token count:"
    print "\tunk:%d\ttotal:%d\tratio:%g" % (test_unk, test_total, 1.0*test_unk/test_total)

def main():
    # get_confusion_matrix()
    # analyze_no_relations()
    analyze_unk()

if __name__ == '__main__':
    main()
