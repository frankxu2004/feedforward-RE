#!/usr/bin/env python
"""Take in data stream of sentences, predict a relation, and output in kbp system-compatible format.
"""

import time
import os
import sys
import json
import random
import tensorflow as tf
import numpy as np

import data_utils
import utils

tf.app.flags.DEFINE_string('data_dir', '../data/basic', 'Directory of the data')
tf.app.flags.DEFINE_string('train_dir', './train/', 'Directory to save training checkpoint files')
tf.app.flags.DEFINE_string('model', 'lstm', 'Must be from lstm, cnn, conv-lstm')
tf.app.flags.DEFINE_integer('batch_size', 50, 'The size of minibatch used for testing.')

FLAGS = tf.app.flags.FLAGS

# correctly import models
if FLAGS.model == 'lstm':
    import model
elif FLAGS.model == 'cnn':
    import cnn_model as model
elif FLAGS.model == 'conv-lstm':
    import conv_rnn_model as model
else:
    raise AttributeError("Model unimplemented: " + FLAGS.model)

def predict():
    # create graph based on model and load trained parameters 
    with tf.Graph().as_default():
        ### the first model will be doing the full batches (a residual of examples will be left)
        with tf.variable_scope('model'):
            m = _get_model(is_train=False)
        saver = tf.train.Saver(tf.all_variables())

        config = tf.ConfigProto()
        gpu_options = tf.GPUOptions(allow_growth=True)
        sess = tf.Session(config=tf.ConfigProto(device_count={"GPU":1}, gpu_options=gpu_options))
        ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
        print >> sys.stderr, "Preparing to load model from " + FLAGS.train_dir
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            raise IOError("Loading checkpoint file failed!")
    
    # load vocab
    word2id = data_utils.load_from_dump(os.path.join(FLAGS.data_dir, '%d.vocab'%FLAGS.vocab_size))
    id2word = dict([(v,k) for k,v in word2id.iteritems()])

    # load label2id mapping and create inverse mapping
    label2id = data_utils.LABEL_TO_ID
    id2label = dict([(v,k) for k,v in label2id.iteritems()])

    # load data from stdin, and run through model
    for line in sys.stdin:
        batch = json.loads(line.strip())
        # batch should be a list of length-10 list:
        # [words, pos_tags, ner_tags, subj_id, obj_id, subj_ner, obj_ner, subj_begin, subj_end, obj_begin, obj_end]
        batch_map = preprocess_batch(batch, word2id)

        # run model through a batch
        if FLAGS.model == 'lstm':
            use_position = (FLAGS.attn and FLAGS.attn_pos_emb > 0)
            feed = _get_feed_dict(m, batch_map, use_pos=(FLAGS.pos_size > 0), use_ner=(FLAGS.ner_size > 0), 
                    use_position=use_position, position_type='zero')
        elif FLAGS.model == 'cnn':
            use_position = (FLAGS.pos_emb_size > 0)
            feed = _get_feed_dict(m, batch_map, use_pos=(FLAGS.pos_size > 0), use_ner=(FLAGS.ner_size > 0),
                    use_position=use_position, position_type='separate')
        predictions, confidences = sess.run([m.prediction, m.confidence], feed_dict=feed)

        outputs = postprocess_batch(predictions, confidences, batch_map, id2label)
        for line in outputs:
            print line

def _get_model(is_train):
    if FLAGS.model == 'lstm':
        return model.LSTMModel(is_train=is_train)
    elif FLAGS.model == 'cnn':
        return model.SimpleCNNModel(is_train=is_train)
    elif FLAGS.model == 'conv-lstm':
        return model.ConvLSTMModel(is_train=is_train)
    else:
        raise AttributeError("Model unimplemented: " + FLAGS.model)

def _get_feed_dict(model, batch_map, use_pos=False, use_ner=False, use_position=False, position_type='zero'):
    feed = {}
    feed[model.word_inputs] = batch_map['words']
    feed[model.seq_lens] = batch_map['seq_len']
    if use_pos:
        feed[model.pos_inputs] = batch_map['pos_tags']
    if use_ner:
        feed[model.ner_inputs] = batch_map['ner_tags']
    if use_position:
        subj_pos, obj_pos = _create_entity_position_seq(batch_map['subj_begin'], batch_map['subj_end'],
                batch_map['obj_begin'], batch_map['obj_end'], batch_map['seq_len'], position_type)
        feed[model.subj_pos_inputs] = subj_pos
        feed[model.obj_pos_inputs] = obj_pos
        #print >> sys.stderr, subj_pos
        #print >> sys.stderr, obj_pos
    return feed

def _create_entity_position_seq(subj_begin, subj_end, obj_begin, obj_end, seq_len, position_type):
    pad_id = 0
    if position_type == 'zero': # for LSTM
        pad_id = 0
    elif position_type == 'separate': # for CNN
        pad_id = -FLAGS.sent_len-1
    else:
        raise Exception('Unsupported position input sequence type: ' + position_type)
    subj_pos_seq = []
    obj_pos_seq = []
    for sb, se, ob, oe, l in zip(subj_begin, subj_end, obj_begin, obj_end, seq_len):
        subj_pos_seq.append(_generate_position_seq(FLAGS.sent_len, sb, se, l, pad_id))
        obj_pos_seq.append(_generate_position_seq(FLAGS.sent_len, ob, oe, l, pad_id))
    return subj_pos_seq, obj_pos_seq

def _generate_position_seq(sent_len, min_ent_idx, max_ent_idx, seq_len, pad_output):
    position_seq = []
    for i in range(sent_len):
        if i >= seq_len:
            position_seq.append(pad_output)
        elif i < min_ent_idx:
            position_seq.append(i-min_ent_idx)
        elif i > max_ent_idx:
            position_seq.append(i-max_ent_idx)
        else:
            position_seq.append(0)
    return position_seq

def preprocess_batch(batch, word2id):
    ''' Preprocess a batch from stdin'''
    output_dict = {'words':[], 'pos_tags':[], 'ner_tags':[], 'seq_len': [], 'subj_id': [], 'obj_id': [], 
            'subj_begin':[], 'subj_end':[], 'obj_begin':[], 'obj_end':[]}
    for p in batch:
        words, pos_tags, ner_tags, subj_id, obj_id, subj_ner, obj_ner = p[0], p[1], p[2], int(p[3]), int(p[4]), p[5], p[6]
        subj_begin, subj_end, obj_begin, obj_end = int(p[7]), int(p[8]), int(p[9]), int(p[10])
        # anonymize words
        for i in range(subj_begin, subj_end):
            words[i] = subj_ner + '-SUBJ'
        for i in range(obj_begin, obj_end):
            words[i] = obj_ner + '-OBJ'
        # convert into word index
        words = [word2id[x.lower()] if x.lower() in word2id else data_utils.UNK_ID for x in words]
        pos_tags = [data_utils.POS_TO_ID[x] if x in data_utils.POS_TO_ID else data_utils.UNK_ID for x in pos_tags]
        ner_tags = [data_utils.NER_TO_ID[x] if x in data_utils.NER_TO_ID else data_utils.UNK_ID for x in ner_tags]
        #print >> sys.stderr, pos_tags
        #print >> sys.stderr, ner_tags
        seq_len = len(words)
        # pad words into max length
        assert(seq_len <= FLAGS.sent_len)
        words += [data_utils.PAD_ID] * (FLAGS.sent_len - seq_len)
        pos_tags += [data_utils.PAD_ID] * (FLAGS.sent_len - seq_len)
        ner_tags += [data_utils.PAD_ID] * (FLAGS.sent_len - seq_len)
        # populate output
        output_dict['words'].append(words)
        output_dict['pos_tags'].append(pos_tags)
        output_dict['ner_tags'].append(ner_tags)
        output_dict['seq_len'].append(seq_len)
        output_dict['subj_id'].append(subj_id)
        output_dict['obj_id'].append(obj_id)
        output_dict['subj_begin'].append(subj_begin)
        output_dict['subj_end'].append(subj_end)
        output_dict['obj_begin'].append(obj_begin)
        output_dict['obj_end'].append(obj_end)
    return output_dict

def postprocess_batch(predictions, confidences, batch_map, id2label):
    outputs = []
    for pred, conf, subj_id, obj_id in zip(predictions, confidences, batch_map['subj_id'], batch_map['obj_id']):
        rel = id2label[pred]
        if rel != 'no_relation':
            outputs.append('%d\t%s\t%d\t%g' % (subj_id, rel, obj_id, conf))
    return outputs

def main(_):
    predict()

if __name__ == '__main__':
    tf.app.run()

