from datetime import datetime
import time
import os
import sys
import random
import tensorflow as tf
import numpy as np

import data_utils
import utils
import scorer

tf.app.flags.DEFINE_string('data_dir', '../data/basic', 'Directory of the data')
tf.app.flags.DEFINE_string('train_dir', './train/', 'Directory to save training checkpoint files')
tf.app.flags.DEFINE_string('eval_set', 'test', 'The dataset used to do evaluation. Default to be test set.')

tf.app.flags.DEFINE_boolean('use_confidence', True, 'Whether to output confidence in prediction file. If False: will use 1.0 for confidence.')
tf.app.flags.DEFINE_string('model', 'lstm', 'Must be from lstm, cnn, conv-lstm')
tf.app.flags.DEFINE_integer('batch_size', 200, 'The size of minibatch used for testing.')
tf.app.flags.DEFINE_float('f_measure', 1.0, 'The f measurement to use. Default to be 1. E.g. f-0.5 will favor precision over recall.')

#tf.app.flags.DEFINE_string('write_prob', False, 'Whether to write the raw probability matrix to file.')
tf.app.flags.DEFINE_string('output_dir', './tmp/', 'Directory to save output files.')
tf.app.flags.DEFINE_boolean('cleanup', False, 'Whether to clean up the temp key and prediction file')
tf.app.flags.DEFINE_float('sample_neg', 1, 'Resampling negative data with this rate. Applied to both train and dev.')

FLAGS = tf.app.flags.FLAGS

#orrectly import models
if FLAGS.model == 'lstm':
    import model
    _get_feed_dict = utils._get_feed_dict_with_position_seq_zero_padding
elif FLAGS.model == 'cnn':
    import cnn_model as model
    _get_feed_dict = utils._get_feed_dict_with_position_seq_separate_padding
elif FLAGS.model == 'conv-lstm':
    import conv_rnn_model as model
    _get_feed_dict = utils._get_feed_dict
else:
    raise AttributeError("Model unimplemented: " + FLAGS.model)

def evaluate():
    print "Building graph and loading model..."
    with tf.Graph().as_default():
        ### the first model will be doing the full batches (a residual of examples will be left)
        with tf.variable_scope('model'):
            m = _get_model(is_train=False)
        saver = tf.train.Saver(tf.all_variables())

        config = tf.ConfigProto()
        gpu_options = tf.GPUOptions(allow_growth=True)
        sess = tf.Session(config=tf.ConfigProto(device_count={"GPU":1}, gpu_options=gpu_options))
        ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            raise IOError("Loading checkpoint file failed!")
        
        print "====> Evaluating on %s data" % FLAGS.eval_set
        print "Loading %s data..." % FLAGS.eval_set
        loader = data_utils.DataLoader(os.path.join(FLAGS.data_dir, '%s.vocab%d.id'%(FLAGS.eval_set, FLAGS.vocab_size)), 
            FLAGS.batch_size, FLAGS.sent_len, shuffle=False, sample_neg=FLAGS.sample_neg) # load test data with batch_size 1; this is too slow

        # load label2id mapping and create inverse mapping
        label2id = data_utils.LABEL_TO_ID
        id2label = dict([(v,k) for k,v in label2id.iteritems()])

        # key = random.randint(1e5, 1e6-1) # get a random 6-digit int
        test_key_file = os.path.join(FLAGS.output_dir, 'shuffled.%s.key.tmp' % FLAGS.eval_set)
        test_prediction_file = os.path.join(FLAGS.output_dir, 'shuffled.%s.prediction.tmp' % FLAGS.eval_set)
        test_prob_file = os.path.join(FLAGS.output_dir, 'shuffled.%s.probs.tmp' % FLAGS.eval_set)
        loader.write_keys(test_key_file, id2label=id2label, include_residual=True) # write shuffled key to file, used by scorer

        test_loss = .0
        print "Evaluating on %d test examples with full batch..." % (loader.num_batches * loader.batch_size)
        preds, confs = [], []
        all_probs = np.zeros([loader.num_examples ,FLAGS.num_class])
        for i in range(loader.num_batches):
            x, y, x_lens = loader.next_batch()
            feed = _get_feed_dict(m, x, y, x_lens, use_pos=(FLAGS.pos_size > 0), use_ner=(FLAGS.ner_size > 0), use_signature=(FLAGS.signature_size > 0))
            loss_value, predictions, confidences, probs = sess.run([m.loss, m.prediction, m.confidence, m.probs], feed_dict=feed)
            test_loss += loss_value
            preds += list(predictions)
            confs += list(confidences)
            all_probs[i*loader.batch_size:(i+1)*loader.batch_size, :] = probs
        
        if loader.num_residual > 0:
            print "Evaluating on an residual of %d examples..." % loader.num_residual
            x, y, x_lens = loader.get_residual()
            feed = _get_feed_dict(m, x, y, x_lens, use_pos=(FLAGS.pos_size > 0), use_ner=(FLAGS.ner_size > 0), use_signature=(FLAGS.signature_size > 0))
            loss_value, predictions, confidences, probs = sess.run([m.loss, m.prediction, m.confidence, m.probs], feed_dict=feed)
            test_loss += loss_value
            preds += list(predictions)
            confs += list(confidences)
            all_probs[loader.num_batches*loader.batch_size:, :] = probs

        if not FLAGS.use_confidence:
            confs = [1.0] * len(confs)

        _write_prediction_file(preds, confs, all_probs, id2label, test_prediction_file, test_prob_file)
        test_loss /= loader.num_examples
        print "%s: test_loss = %.6f" % (datetime.now(), test_loss)

        prec, recall, f = scorer.score(test_key_file, [test_prediction_file], FLAGS.f_measure, verbose=True)

    # clean up
    if FLAGS.cleanup and os.path.exists(test_key_file):
        os.remove(test_key_file)
    if FLAGS.cleanup and os.path.exists(test_prediction_file):
        os.remove(test_prediction_file)

def _write_prediction_file(preds, confs, all_probs, id2label, pred_file, prob_file):
    assert len(preds) == len(confs)
    with open(pred_file, 'w') as outfile:
        for p, c in zip(preds, confs):
            outfile.write(str(id2label[p]) + '\t' + str(c) + '\n')
    with open(prob_file, 'w') as outfile:
        for i, p in enumerate(preds):
            probs = all_probs[i,:].tolist()
            probs_str = [str(x) for x in probs]
            outfile.write(str(id2label[p]) + '\t')
            outfile.write('\t'.join(probs_str) + '\n')
    return

def _get_model(is_train):
    if FLAGS.model == 'lstm':
        return model.LSTMModel(is_train=is_train)
    elif FLAGS.model == 'cnn':
        return model.SimpleCNNModel(is_train=is_train)
    elif FLAGS.model == 'conv-lstm':
        return model.ConvLSTMModel(is_train=is_train)
    else:
        raise AttributeError("Model unimplemented: " + FLAGS.model)

def main(_):
    evaluate()

if __name__ == '__main__':
    tf.app.run()
