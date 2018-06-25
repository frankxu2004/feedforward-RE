import os
import sys
import time
import numpy as np
import random
import cPickle as pickle
from collections import Counter, OrderedDict
from stanza.text.dataset import Dataset

PAD_TOKEN = '<PAD>'
UNK_TOKEN = '<UNK>'
PAD_ID = 0
UNK_ID = 1

# use the count of words to build vocab, instead of using a fixed-size vocab
MIN_WORD_COUNT = 2
USE_COUNT = True

VOCAB_SIZE = 30000 # 30000 seems to have a good balance
DATA_ROOT = "../data/"

# data files names
TRAIN_FILE = DATA_ROOT + 'tacred/train.anon-direct.conll'
DEV_FILE = DATA_ROOT + 'tacred/dev.anon-direct.conll'
TEST_FILE = DATA_ROOT + 'tacred/test.anon-direct.conll'

#LABEL2ID_FILE = DATA_ROOT + 'basic/label2id.dict'

# hard-coded mappings from fields to ids
SUBJ_NER_TO_ID = {PAD_TOKEN: 0, UNK_TOKEN: 1, 'ORGANIZATION': 2, 'PERSON': 3}
OBJ_NER_TO_ID = {PAD_TOKEN: 0, UNK_TOKEN: 1, 'PERSON': 2, 'ORGANIZATION': 3, 'DATE': 4, 'NUMBER': 5, 'TITLE': 6, 'COUNTRY': 7, 'LOCATION': 8, 'CITY': 9, 'MISC': 10, 'STATE_OR_PROVINCE': 11, 'DURATION': 12, 'NATIONALITY': 13, 'CAUSE_OF_DEATH': 14, 'CRIMINAL_CHARGE': 15, 'RELIGION': 16, 'URL': 17, 'IDEOLOGY': 18}
NER_TO_ID = {PAD_TOKEN: 0, UNK_TOKEN: 1, 'O': 2, 'PERSON': 3, 'ORGANIZATION': 4, 'LOCATION': 5, 'DATE': 6, 'NUMBER': 7, 'MISC': 8, 'DURATION': 9, 'MONEY': 10, 'PERCENT': 11, 'ORDINAL': 12, 'TIME': 13, 'SET': 14}
POS_TO_ID = {PAD_TOKEN: 0, UNK_TOKEN: 1, 'NNP': 2, 'NN': 3, 'IN': 4, 'DT': 5, ',': 6, 'JJ': 7, 'NNS': 8, 'VBD': 9, 'CD': 10, 'CC': 11, '.': 12, 'RB': 13, 'VBN': 14, 'PRP': 15, 'TO': 16, 'VB': 17, 'VBG': 18, 'VBZ': 19, 'PRP$': 20, ':': 21, 'POS': 22, '\'\'': 23, '``': 24, '-RRB-': 25, '-LRB-': 26, 'VBP': 27, 'MD': 28, 'NNPS': 29, 'WP': 30, 'WDT': 31, 'WRB': 32, 'RP': 33, 'JJR': 34, 'JJS': 35, '$': 36, 'FW': 37, 'RBR': 38, 'SYM': 39, 'EX': 40, 'RBS': 41, 'WP$': 42, 'PDT': 43, 'LS': 44, 'UH': 45, '#': 46}
DEPREL_TO_ID = {PAD_TOKEN: 0, UNK_TOKEN: 1, 'punct': 2, 'compound': 3, 'case': 4, 'nmod': 5, 'det': 6, 'nsubj': 7, 'amod': 8, 'conj': 9, 'dobj': 10, 'ROOT': 11, 'cc': 12, 'nmod:poss': 13, 'mark': 14, 'advmod': 15, 'appos': 16, 'nummod': 17, 'dep': 18, 'ccomp': 19, 'aux': 20, 'advcl': 21, 'acl:relcl': 22, 'xcomp': 23, 'cop': 24, 'acl': 25, 'auxpass': 26, 'nsubjpass': 27, 'nmod:tmod': 28, 'neg': 29, 'compound:prt': 30, 'mwe': 31, 'parataxis': 32, 'root': 33, 'nmod:npmod': 34, 'expl': 35, 'csubj': 36, 'cc:preconj': 37, 'iobj': 38, 'det:predet': 39, 'discourse': 40, 'csubjpass': 41}

LABEL_TO_ID = {'no_relation': 0, 'per:title': 1, 'org:top_members/employees': 2, 'per:employee_of': 3, 'org:alternate_names': 4, 'org:country_of_headquarters': 5, 'per:countries_of_residence': 6, 'org:city_of_headquarters': 7, 'per:cities_of_residence': 8, 'per:age': 9, 'per:stateorprovinces_of_residence': 10, 'per:origin': 11, 'org:subsidiaries': 12, 'org:parents': 13, 'per:spouse': 14, 'org:stateorprovince_of_headquarters': 15, 'per:children': 16, 'per:other_family': 17, 'per:alternate_names': 18, 'org:members': 19, 'per:siblings': 20, 'per:schools_attended': 21, 'per:parents': 22, 'per:date_of_death': 23, 'org:member_of': 24, 'org:founded_by': 25, 'org:website': 26, 'per:cause_of_death': 27, 'org:political/religious_affiliation': 28, 'org:founded': 29, 'per:city_of_death': 30, 'org:shareholders': 31, 'org:number_of_employees/members': 32, 'per:date_of_birth': 33, 'per:city_of_birth': 34, 'per:charges': 35, 'per:stateorprovince_of_death': 36, 'per:religion': 37, 'per:stateorprovince_of_birth': 38, 'per:country_of_birth': 39, 'org:dissolved': 40, 'per:country_of_death': 41}

# field names
WORD_FIELD = 'word'
SUBJ_FIELD = 'subj'
OBJ_FIELD = 'obj'
SUBJ_NER_FIELD = 'subj_ner'
OBJ_NER_FIELD = 'obj_ner'
POS_FIELD = 'stanford_pos'
NER_FIELD = 'stanford_ner'

LABLE_FIELD = 'label'

MAX_SENT_LENGTH = 96

np.random.seed(1234)

def load_datasets(fnames, lowercase=True):
    datasets = []
    for fn in fnames:
        d = Dataset.load_conll(fn)
        print "\t%d examples in %s" % (len(d), fn)
        if lowercase:
            converters = {'word': lambda word_list: [x.lower() if x is not None else None for x in word_list]}
            d.convert(converters, in_place=True)
        datasets.append(d)
    return datasets

def build_vocab(datasets, use_count):
    c = Counter()
    for d in datasets:
        for row in d:
            c.update(row['word'])
    print "%d tokens found in dataset." % len(c)
    if use_count:
        print "Use min word count threshold to create vocab."
        all_words = [k for k,v in c.items() if v >= MIN_WORD_COUNT]
        word_id_pairs = [(x, i+2) for i, x in enumerate(all_words)]
    else:
        print "Use fixed vocab size of " + str(VOCAB_SIZE)
        word_id_pairs = [(x[0], i+2) for i, x in enumerate(c.most_common(VOCAB_SIZE - 2))]
    word_id_pairs += [(UNK_TOKEN, UNK_ID), (PAD_TOKEN, PAD_ID)]
    word2id = dict(word_id_pairs)
    return word2id

def convert_words_to_ids(datasets, word2id):
    new_datasets = []
    for d in datasets:
        for i,row in enumerate(d):
            tokens = row['word']
            tokens_by_ids = [word2id[x] if x in word2id else UNK_ID for x in tokens]
            d.fields['word'][i] = tokens_by_ids
        new_datasets.append(d)
    return new_datasets

def get_label_to_id(datasets):
    c = Counter()
    for d in datasets:
        c.update(d.fields['label'])
    print "\t %d labels found, converted to ids." % (len(c))
    label2id = dict([(x[0],i) for i,x in enumerate(c.most_common())])
    return label2id

def convert_fields_to_ids(datasets, field2map):
    new_datasets = []
    for d in datasets:
        for f,m in field2map.iteritems(): # f is fieldname, m is the field2id map
            for i,l in enumerate(d.fields[f]):
                if isinstance(l, list): # the field could be list, in case like NER
                    for j, w in enumerate(l):
                        if w is not None: # we skip None in some fields like subj_ner
                            d.fields[f][i][j] = m[w]
                else: # or can be just a label
                    d.fields[f][i] = m[l]
        new_datasets.append(d)
    return new_datasets

def preprocess():
    print "Loading data from files..."
    d_train, d_dev, d_test = load_datasets([TRAIN_FILE, DEV_FILE, TEST_FILE])
    print "Build vocab from training and dev set..."
    word2id = build_vocab([d_train], USE_COUNT)
    VOCAB_SIZE = len(word2id)
    vocab_file = os.path.join(DATA_ROOT, "basic/%d.vocab" % VOCAB_SIZE)
    dump_to_file(vocab_file, word2id)
    print "Vocab with %d words saved to file %s" % (len(word2id), vocab_file)
    
    #print "Collecting labels..."
    #label2id = get_label_to_id([d_train])
    #dump_to_file(LABEL2ID_FILE, label2id)

    print "Converting data to ids..."
    d_train, d_dev, d_test = convert_words_to_ids([d_train, d_dev, d_test], word2id)
    d_train, d_dev, d_test = convert_fields_to_ids([d_train, d_dev, d_test], \
            {'label': LABEL_TO_ID, 'stanford_pos': POS_TO_ID, 'stanford_ner': NER_TO_ID, 'subj_ner': SUBJ_NER_TO_ID, 'obj_ner': OBJ_NER_TO_ID})

    # generate file names
    TRAIN_ID_FILE = DATA_ROOT + 'basic/train.vocab%d.id' % VOCAB_SIZE
    DEV_ID_FILE = DATA_ROOT + 'basic/dev.vocab%d.id' % VOCAB_SIZE
    TEST_ID_FILE = DATA_ROOT + 'basic/test.vocab%d.id' % VOCAB_SIZE
    dump_to_file(TRAIN_ID_FILE, d_train)
    dump_to_file(DEV_ID_FILE, d_dev)
    dump_to_file(TEST_ID_FILE, d_test)
    print "Datasets saved to files."
    max_length = 0
    for d in [d_train, d_dev, d_test]:
        for row in d:
            l = len(row['word'])
            if l > max_length:
                max_length = l
    print "Datasets maximum sentence length is %d." % max_length

class DataLoader():
    def __init__(self, dump_name, batch_size, pad_len, shuffle=True, subsample=1, unk_prob=0, sample_neg=1.0):
        self.dataset = load_from_dump(dump_name)
        if shuffle:
            self.dataset = self.dataset.shuffle()
        if subsample < 1:
            n = int(subsample * len(self.dataset))
            self.dataset = Dataset(self.dataset[:n])
        if sample_neg != 1.0:
            if sample_neg <= 0 or sample_neg >= 2:
                raise Exception("Invalid negative resampling rate: " + str(sample_neg))
            self.dataset = self.create_new_by_resample_neg(self.dataset, sample_neg)
        if shuffle:
            self.dataset = self.dataset.shuffle()
        self.batch_size = batch_size
        self.num_examples = len(self.dataset)
        self.num_batches = self.num_examples // self.batch_size
        self.num_residual = self.num_examples - self.batch_size * self.num_batches
        self.pad_len = pad_len
        self._unk_prob = unk_prob
        self._pointer = 0

    def next_batch(self):
        """
        Generate the most simple batch. x_batch is sentences, y_batch is labels, and x_lens is the unpadded length of sentences in x_batch.
        """
        x_batch = {WORD_FIELD: [], POS_FIELD: [], NER_FIELD: [], SUBJ_FIELD: [], OBJ_FIELD: [], SUBJ_NER_FIELD: [], OBJ_NER_FIELD: []}
        x_lens = []
        for field in x_batch.keys():
            for tokens in self.dataset.fields[field][self._pointer:self._pointer + self.batch_size]:
                if field == WORD_FIELD: # we need to 1) corrupt 2) count sent len with word field
                    if self._unk_prob > 0:
                        tokens = self.corrupt_sentence(tokens)
                    x_lens.append(len(tokens))
                # apply padding to the left
                assert self.pad_len >= len(tokens), "Padding length is shorter than original sentence length."
                tokens = tokens + [PAD_ID] * (self.pad_len - len(tokens))
                x_batch[field].append(tokens)
        y_batch = self.dataset.fields[LABLE_FIELD][self._pointer:self._pointer + self.batch_size]
        self._pointer += self.batch_size
        return x_batch, y_batch, x_lens

    def get_residual(self):
        x_batch = {WORD_FIELD: [], POS_FIELD: [], NER_FIELD: [], SUBJ_FIELD: [], OBJ_FIELD: []}
        x_lens = []
        for field in x_batch.keys():
            for tokens in self.dataset.fields[field][self._pointer:]:
                if field == WORD_FIELD: # we need to 1) corrupt 2) count sent len with word field
                    if self._unk_prob > 0:
                        tokens = self.corrupt_sentence(tokens)
                    x_lens.append(len(tokens))
                tokens = tokens + [PAD_ID] * (self.pad_len - len(tokens))
                x_batch[field].append(tokens)
        y_batch = self.dataset.fields[LABLE_FIELD][self._pointer:]
        return x_batch, y_batch, x_lens

    def reset_pointer(self):
        self._pointer = 0

    def corrupt_sentence(self, tokens):
        new_tokens = []
        for x in tokens:
            if x != UNK_ID and np.random.random() < self._unk_prob:
                new_tokens.append(UNK_ID)
            else:
                new_tokens.append(x)
        return new_tokens

    def add_to_new_dataset(self, new_dataset, row):
        for k in row.keys():
            new_dataset[k].append(row[k])

    def create_new_by_resample_neg(self, dataset, neg_sample_rate):
        new_dataset = OrderedDict()
        for k in dataset.fields.keys():
            new_dataset[k] = []
        # start resampling
        for i, row in enumerate(dataset):
            if row[LABLE_FIELD] == LABEL_TO_ID['no_relation']:
                if neg_sample_rate < 1.0 and random.random() <= neg_sample_rate:
                    # keep this negative example
                    self.add_to_new_dataset(new_dataset, row)
                elif neg_sample_rate >= 1.0:
                    # first keep
                    self.add_to_new_dataset(new_dataset, row)
                    if random.random() <= (neg_sample_rate - 1.0):
                        # then decide whether to repeat
                        self.add_to_new_dataset(new_dataset, row)
            else: # keep all non-negative examples
                self.add_to_new_dataset(new_dataset, row)
        new_dataset = Dataset(new_dataset)
        print >> sys.stderr, "New dataset created by resampling negative: %d examples before, %d (%g) examples after." % (len(dataset), len(new_dataset), \
            float(len(new_dataset)) / len(dataset))
        return new_dataset

    def write_keys(self, key_file, id2label=None, include_residual=False):
        if id2label is None:
            id2label = lambda x: x # map to itself
        if include_residual:
            end_index = self.num_examples
        else:
            end_index = self.num_batches * self.batch_size
        labels = [id2label[l] for l in self.dataset.fields[LABLE_FIELD][:end_index]]
        # write to file
        with open(key_file, 'w') as outfile:
            for l in labels:
                outfile.write(str(l) + '\n')
        return

def dump_to_file(filename, obj):
    with open(filename, 'wb') as outfile:
        pickle.dump(obj, file=outfile)
    return

def load_from_dump(filename):
    with open(filename, 'rb') as infile:
        obj = pickle.load(infile)
    return obj

def test():
    print "Loading training data from files..."
    train_loader = DataLoader(os.path.join(DATA_ROOT, 'basic/train.vocab36000.id'), 
        50, MAX_SENT_LENGTH, subsample=1, unk_prob=0.04)
    print "Data loaded. Start batching..."
    start_time = time.time()
    num_epoch = 100
    for i in range(num_epoch):
        train_loader.reset_pointer()
        for _ in range(train_loader.num_batches):
            batch = train_loader.next_batch()
        batch = train_loader.get_residual()
        #print batch[0][SUBJ_FIELD]
    print "Batching time for %d epochs: %d seconds." % (num_epoch, time.time() - start_time)

def main():
    preprocess()
    #test()

if __name__ == '__main__':
    main()
