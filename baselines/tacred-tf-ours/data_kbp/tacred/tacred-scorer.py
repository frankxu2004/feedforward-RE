#!/usr/bin/env python
#

import argparse
import sys
from collections import Counter

def parse_arguments():
    """
    Parse the command line arguments for the program.

    @return An object of parsed arguments from an AgumentParser object.
    """
    parser = argparse.ArgumentParser(description='Score predictions from the TAC-RE dataset')
    parser.add_argument('-key',         nargs=1,   action='store', dest='key',          help='The answer key; one relation per line')
    parser.add_argument('-predictions', nargs='+', action='store', dest='prediction',   help='A list of prediction files; a TSV of "prediction \\t confidence" lines')
    parser.add_argument('-verbose',                action='store_true', dest='verbose', help='If true, print out detailed statistics')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    # Parse the arguments from stdin
    args = parse_arguments();
    key = [str(line).rstrip('\n') for line in open(str(args.key[0]))]
    predictions = [[str(line).rstrip('\n').split('\t') for line in open(str(prediction))] for prediction in args.prediction]

    # Check that the lengths match
    for prediction in predictions:
        if len(prediction) != len(key):
            print("Key and prediction file must have same number of elements: %d in key vs %d in prediction" % (len(key), len(prediction)))
            quit(1)


    # The sufficient statistics for computing accuracy scores
    correct_by_relation = Counter()
    guessed_by_relation = Counter()
    gold_by_relation    = Counter()

    # Loop over the data to compute a score
    for row in range(len(key)):
        gold = key[row]
        guess = 'no_relation'
        guess_score = -1.0;
        for system in predictions:
            system_guess = system[row][0]
            system_score = float(system[row][1])
            if system_guess != 'no_relation' and system_score > guess_score:
                guess = system_guess
                guess_score = system_score

        if gold == 'no_relation' and guess == 'no_relation':
            pass
        elif gold == 'no_relation' and guess != 'no_relation':
            guessed_by_relation[guess] += 1
        elif gold != 'no_relation' and guess == 'no_relation':
            gold_by_relation[gold] += 1
        elif gold != 'no_relation' and guess != 'no_relation':
            guessed_by_relation[guess] += 1
            gold_by_relation[gold] += 1
            if gold == guess:
                correct_by_relation[guess] += 1

    # Print verbose information
    if args.verbose:
        print "Per-relation statistics:"
        relations = gold_by_relation.keys()
        longest_relation = 0
        for relation in sorted(relations):
            longest_relation = max(len(relation), longest_relation)
        for relation in sorted(relations):
            # (compute the score)
            correct = correct_by_relation[relation]
            guessed = guessed_by_relation[relation]
            gold    = gold_by_relation[relation]
            prec = 1.0
            if guessed > 0:
                prec = float(correct) / float(guessed)
            recall = 0.0
            if gold > 0:
                recall = float(correct) / float(gold)
            f1 = 0.0
            if prec + recall > 0:
                f1 = 2.0 * prec * recall / (prec + recall)
            # (print the score)
            sys.stdout.write(("{:<" + str(longest_relation) + "}").format(relation))
            sys.stdout.write("  P: ")
            if prec < 0.1: sys.stdout.write(' ')
            if prec < 1.0: sys.stdout.write(' ')
            sys.stdout.write("{:.2%}".format(prec))
            sys.stdout.write("  R: ")
            if recall < 0.1: sys.stdout.write(' ')
            if recall < 1.0: sys.stdout.write(' ')
            sys.stdout.write("{:.2%}".format(recall))
            sys.stdout.write("  F1: ")
            if f1 < 0.1: sys.stdout.write(' ')
            if f1 < 1.0: sys.stdout.write(' ')
            sys.stdout.write("{:.2%}".format(f1))
            sys.stdout.write("  #: %d" % gold)
            sys.stdout.write("\n")
        print ""


    # Print the aggregate score
    if args.verbose:
        print "Final Score:"
    prec_micro = 1.0
    if sum(guessed_by_relation.values()) > 0:
        prec_micro   = float(sum(correct_by_relation.values())) / float(sum(guessed_by_relation.values()))
    recall_micro = 0.0
    if sum(gold_by_relation.values()) > 0:
        recall_micro = float(sum(correct_by_relation.values())) / float(sum(gold_by_relation.values()))
    f1_micro = 0.0
    if prec_micro + recall_micro > 0.0:
        f1_micro = 2.0 * prec_micro * recall_micro / (prec_micro + recall_micro)
    print( "Precision (micro): {:.3%}".format(prec_micro) )
    print( "   Recall (micro): {:.3%}".format(recall_micro) )
    print( "       F1 (micro): {:.3%}".format(f1_micro) )
