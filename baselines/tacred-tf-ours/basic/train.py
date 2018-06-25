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
tf.app.flags.DEFINE_string('train_dir', './train/test', 'Directory to save training checkpoint files')
tf.app.flags.DEFINE_integer('num_epoch', 50, 'Number of epochs to run')
tf.app.flags.DEFINE_integer('batch_size', 50, 'The size of minibatch used for training.')
tf.app.flags.DEFINE_boolean('use_pretrain', False, 'Use pretrained embeddings or not')
tf.app.flags.DEFINE_float('corrupt_rate', 0.04, 'The rate at which we corrupt training data with UNK token.')

tf.app.flags.DEFINE_string('model', 'lstm', 'Must be from lstm, cnn, conv-lstm')

tf.app.flags.DEFINE_float('init_lr', 0.5, 'Initial learning rate')
tf.app.flags.DEFINE_float('lr_decay', 0.9, 'LR decay rate')

tf.app.flags.DEFINE_boolean('cont_train', False, 'Whether to continue training using the model saved in the training dir.')
tf.app.flags.DEFINE_integer('log_step', 10, 'Write log to stdout after this step')
# tf.app.flags.DEFINE_integer('summary_step', 200, 'Write summary after this step')
# tf.app.flags.DEFINE_integer('save_epoch', 5, 'Save model after this epoch') # save the model that achieves the best eval f1
tf.app.flags.DEFINE_float('f_measure', 1.0, 'The f measurement to use. Default to be 1. E.g. f-0.5 will favor precision over recall.')

tf.app.flags.DEFINE_float('gpu_mem', 0.5, 'The fraction of gpu memory to occupy for training')
tf.app.flags.DEFINE_float('subsample', 1, 'The fraction of the training data that are used. 1 means all training data.')
tf.app.flags.DEFINE_float('sample_neg', 1, 'Resampling negative data with this rate. Applied to both train and dev.')
tf.app.flags.DEFINE_integer('seed', 1234, 'Random seed to use.')

FLAGS = tf.app.flags.FLAGS

random.seed(FLAGS.seed)
tf.set_random_seed(FLAGS.seed)

# correctly import models
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

def train():
    # print training params
    print >> sys.stderr, _get_training_info()

    # dealing with files
    print "Loading data from files..."
    train_loader = data_utils.DataLoader(os.path.join(FLAGS.data_dir, 'train.vocab%d.id'%FLAGS.vocab_size), 
        FLAGS.batch_size, FLAGS.sent_len, subsample=FLAGS.subsample, unk_prob=FLAGS.corrupt_rate, sample_neg=FLAGS.sample_neg) # use a subsample of the data if specified
    dev_loader = data_utils.DataLoader(os.path.join(FLAGS.data_dir, 'dev.vocab%d.id'%FLAGS.vocab_size), 
        FLAGS.batch_size, FLAGS.sent_len, sample_neg=FLAGS.sample_neg)
    max_steps = train_loader.num_batches * FLAGS.num_epoch
    
    print "# Examples in training data:"
    print train_loader.num_examples

    # load label2id mapping and create inverse mapping
    label2id = data_utils.LABEL_TO_ID
    id2label = dict([(v,k) for k,v in label2id.iteritems()])

    key = random.randint(1e5, 1e6-1) # get a random 6-digit int
    dev_key_file = os.path.join(FLAGS.train_dir, str(key) + '.shuffled.dev.key.tmp')
    dev_prediction_file = os.path.join(FLAGS.train_dir, str(key) + '.shuffled.dev.prediction.tmp')
    dev_loader.write_keys(dev_key_file, id2label=id2label)

    with tf.Graph().as_default():
        print "Constructing model %s..." % (FLAGS.model)
        with tf.variable_scope('model', reuse=None):
            m = _get_model(is_train=True)
        with tf.variable_scope('model', reuse=True):
            mdev = _get_model(is_train=False)

        saver = tf.train.Saver(tf.all_variables(), max_to_keep=2)
        save_path = os.path.join(FLAGS.train_dir, 'model.ckpt')

        config = tf.ConfigProto()
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=FLAGS.gpu_mem, allow_growth=True)
        sess = tf.Session(config=tf.ConfigProto(device_count={"GPU":1}, gpu_options=gpu_options))
        summary_writer = tf.summary.FileWriter(FLAGS.train_dir, graph=sess.graph)
        sess.run(tf.initialize_all_variables())

        if FLAGS.use_pretrain:
            print "Use pretrained embeddings to initialize model ..."
            emb_file = os.path.join(FLAGS.data_dir, "emb-v%d-d%d.npy" % (FLAGS.vocab_size, FLAGS.hidden_size))
            if not os.path.exists(emb_file):
                raise Exception("Pretrained vector file does not exist at: " + emb_file)
            pretrained_embedding = np.load(emb_file)
            if FLAGS.model == 'cnn': # For CNN, we need to remove the first padding dimension (otherwise we cannot tf.assign it)
                pretrained_embedding = pretrained_embedding[1:,:]
            m.assign_embedding(sess, pretrained_embedding)

        current_lr = FLAGS.init_lr
        global_step = 0
        training_history = []
        dev_f_history = []
        best_dev_scores = []

        def eval_once(mdev, sess, data_loader):
            data_loader.reset_pointer()
            predictions = []
            confidences = []
            dev_loss = 0.0
            for _ in xrange(data_loader.num_batches):
                x_batch, y_batch, x_lens = data_loader.next_batch()
                feed = _get_feed_dict(mdev, x_batch, y_batch, x_lens, use_pos=(FLAGS.pos_size > 0), use_ner=(FLAGS.ner_size > 0), use_signature=(FLAGS.signature_size > 0))
                loss_value, pred, conf = sess.run([mdev.loss, mdev.prediction, mdev.confidence], feed_dict=feed)
                predictions += list(pred)
                confidences += list(conf)
                dev_loss += loss_value
            dev_loss /= data_loader.num_batches
            return dev_loss, predictions, confidences

        print "Start training with %d epochs, and %d steps per epoch..." % (FLAGS.num_epoch, train_loader.num_batches)
        for epoch in xrange(FLAGS.num_epoch):
            train_loss = 0.0
            train_loader.reset_pointer()
            m.assign_lr(sess, current_lr)
            for _ in xrange(train_loader.num_batches):
                global_step += 1
                start_time = time.time()
                x_batch, y_batch, x_lens = train_loader.next_batch()
                feed = _get_feed_dict(m, x_batch, y_batch, x_lens, use_pos=(FLAGS.pos_size > 0), use_ner=(FLAGS.ner_size > 0), use_signature=(FLAGS.signature_size > 0))
                _, loss_value = sess.run([m.train_op, m.loss], feed_dict=feed)
                duration = time.time() - start_time
                train_loss += loss_value
                assert not np.isnan(loss_value), "Model loss is NaN."

                if global_step % FLAGS.log_step == 0:
                    format_str = ('%s: step %d/%d (epoch %d/%d), loss = %.6f (%.3f sec/batch), lr: %.6f')
                    print format_str % (datetime.now(), global_step, max_steps, epoch+1, FLAGS.num_epoch, 
                        loss_value, duration, current_lr)

            # summary loss after each epoch
            train_loss /= train_loader.num_batches
            summary_writer.add_summary(_summary_for_scalar('eval/training_loss', train_loss), global_step=epoch)

            dev_loss, dev_preds, dev_confs = eval_once(mdev, sess, dev_loader)
            summary_writer.add_summary(_summary_for_scalar('eval/dev_loss', dev_loss), global_step=epoch)
            _write_prediction_file(dev_preds, dev_confs, id2label, dev_prediction_file)

            print "Evaluating on dev set..."
            dev_prec, dev_recall, dev_f = scorer.score(dev_key_file, [dev_prediction_file], FLAGS.f_measure)
            print "Epoch %d: training_loss = %.6f" % (epoch+1, train_loss)
            print "Epoch %d: dev_loss = %.6f, dev_f-%g = %.6f" % (epoch+1, dev_loss, FLAGS.f_measure, dev_f)

            # change learning rate if training loss is higher than most recent 3 epochs
            # if len(training_history) > 3 and train_loss >= max(training_history[-3:]):
            # decrease learning rate if dev_f does not increase after an epoch
            if len(dev_f_history) > 20 and dev_f <= dev_f_history[-1]:
                current_lr *= FLAGS.lr_decay
            training_history.append(train_loss)

            # save the model when best f1 is achieved on dev set
            if len(dev_f_history) == 0 or (len(dev_f_history) > 0 and dev_f > max(dev_f_history)):
                saver.save(sess, save_path, global_step=epoch)
                print "\tmodel saved at epoch %d, with best dev dataset f-%g score %.6f" % (epoch+1, FLAGS.f_measure, dev_f)
                best_dev_scores = [dev_prec, dev_recall, dev_f]
            dev_f_history.append(dev_f)

            # stop learning if lr is too low
            if current_lr < 1e-6: break
        # saver.save(sess, save_path, global_step=epoch)
        print "Training ended with %d epochs." % epoch
        print "\tBest dev scores achieved (P, R, F-%g):\t%.3f\t%.3f\t%.3f" % tuple([FLAGS.f_measure] + [x*100 for x in best_dev_scores])

    # clean up
    if os.path.exists(dev_key_file):
        os.remove(dev_key_file)
    if os.path.exists(dev_prediction_file):
        os.remove(dev_prediction_file)

def _get_training_info():
    info = "Training params:\n"
    info += "\topt: %s\n" % FLAGS.opt
    info += "\tinit_lr: %g\n" % FLAGS.init_lr
    info += "\tnum_epoch: %d\n" % FLAGS.num_epoch
    info += "\tbatch_size: %d\n" % FLAGS.batch_size
    info += "\tsent_len: %d\n" % FLAGS.sent_len
    info += "\tcorrupt_rate: %g\n" % FLAGS.corrupt_rate
    info += "\tsubsample: %g\n" % FLAGS.subsample
    info += "\tsample_neg: %g\n" % FLAGS.sample_neg
    info += "\tuse_pretrain: %s\n" % str(FLAGS.use_pretrain)
    info += "\tf_measure: %g\n" % FLAGS.f_measure
    return info

def _get_model(is_train):
    if FLAGS.model == 'lstm':
        return model.LSTMModel(is_train=is_train)
    elif FLAGS.model == 'cnn':
        return model.SimpleCNNModel(is_train=is_train)
    elif FLAGS.model == 'conv-lstm':
        return model.ConvLSTMModel(is_train=is_train)
    else:
        raise AttributeError("Model unimplemented: " + FLAGS.model)

def _summary_for_scalar(name, value):
    return tf.Summary(value=[tf.Summary.Value(tag=name, simple_value=value)])

def _write_prediction_file(preds, confs, id2label, pred_file):
    assert len(preds) == len(confs)
    with open(pred_file, 'w') as outfile:
        for p, c in zip(preds, confs):
            outfile.write(str(id2label[p]) + '\t' + str(c) + '\n')
    return

def main(argv=None):
    dir_exists = tf.gfile.Exists(FLAGS.train_dir)
    if not FLAGS.cont_train:
        if dir_exists:
            print "Directory %s exists, deleting in 5 seconds..." % FLAGS.train_dir
            time.sleep(5)
            tf.gfile.DeleteRecursively(FLAGS.train_dir)
        tf.gfile.MakeDirs(FLAGS.train_dir)
        train()
    else:
        if not dir_exists:
            print '"cont" flag is set to be True, however training dir does not exist. Start from scratch.'
        train()

if __name__ == '__main__':
    tf.app.run()
