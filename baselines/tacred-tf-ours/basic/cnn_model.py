import sys
import tensorflow as tf

import data_utils

# cnn parameters
tf.app.flags.DEFINE_integer('hidden_size', 200, 'Size of word embeddings and hidden layers')
tf.app.flags.DEFINE_integer('pos_size', 0, 'Size of embeddings for POS tags')
tf.app.flags.DEFINE_integer('ner_size', 0, 'Size of embeddings for NER tags')
tf.app.flags.DEFINE_integer('signature_size', 0, 'Size of added signature embeddings into the final representation. 0 = not add.')
# positional cnn parameters
tf.app.flags.DEFINE_integer('pos_emb_size', 0, 'Position embedding size')

tf.app.flags.DEFINE_integer('num_filter', 500, 'Number of filters for each window size')
tf.app.flags.DEFINE_integer('min_window', 2, 'Minimum size of filter window')
tf.app.flags.DEFINE_integer('max_window', 5, 'Maximum size of filter window')

tf.app.flags.DEFINE_integer('vocab_size', 35888, 'Vocabulary size')
tf.app.flags.DEFINE_integer('num_class', 42, 'Number of class to consider')
tf.app.flags.DEFINE_integer('sent_len', 96, 'Input sentence length. This is after the padding is performed.')
tf.app.flags.DEFINE_float('dropout', 0.5, 'Dropout rate that applies to the CNN. 0 is no dropout.')
tf.app.flags.DEFINE_float('l2_reg', 0.001, 'l2 regularization weight')

tf.app.flags.DEFINE_string('opt', 'adagrad', 'The optimizer to use, must be one of adagrad, sgd, adam.')

FLAGS = tf.app.flags.FLAGS
# Vocab size for different fields (other than 'word')
NUM_POS = len(data_utils.POS_TO_ID)
NUM_NER = len(data_utils.NER_TO_ID)
NUM_SUBJ_NER = len(data_utils.SUBJ_NER_TO_ID)
NUM_OBJ_NER = len(data_utils.OBJ_NER_TO_ID)

def _variable_on_cpu(name, shape, initializer, trainable=True):
    with tf.device('/cpu:0'):
        var = tf.get_variable(name, shape, initializer=initializer, trainable=trainable)
    return var

def _variable_cpu_with_weight_decay(name, shape, initializer, wd):
    var = _variable_on_cpu(name, shape, initializer)
    if wd is not None and wd != 0.:
        weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
    else:
        weight_decay = tf.constant(0.0, dtype=tf.float32)
    return var, weight_decay

def _get_optimizer(lr):
    if FLAGS.opt == 'adagrad':
        return tf.train.AdagradOptimizer(lr)
    elif FLAGS.opt == 'sgd':
        return tf.train.GradientDescentOptimizer(lr)
    elif FLAGS.opt == 'adam':
        return tf.train.AdamOptimizer(lr)
    else:
        raise Exception("Unsupported optimizer: " + FLAGS.opt)

def _get_graph_info():
    info = 'Building 1D-CNN with window size %d-%d, %d filters per window, [%d word emb, %d POS emb, %d NER emb], and %d position emb size...' % \
            (FLAGS.min_window, FLAGS.max_window, FLAGS.num_filter, FLAGS.hidden_size, FLAGS.pos_size, FLAGS.ner_size, FLAGS.pos_emb_size)
    return info

def _create_embedding(vocab_size, hidden_size, emb_name):
    ''' Return emb and the trainable part of emb. '''
    emb_pad = _variable_on_cpu(name= emb_name+'_pad', shape=[1,hidden_size],
            initializer=tf.constant_initializer(0.0), trainable=False)
    emb_rest = _variable_on_cpu(name=emb_name+'_rest', shape=[vocab_size-1, hidden_size], 
            initializer=tf.random_uniform_initializer(minval=-1.0, maxval=1.0), trainable=True)
    emb = tf.concat([emb_pad, emb_rest], 0, name=emb_name)
    #emb = _variable_on_cpu(name=emb_name, shape=[vocab_size, hidden_size],
    #        initializer=tf.random_uniform_initializer(minval=-1.0, maxval=1.0), trainable=True)
    return emb, emb_rest

def _create_embedding_layers(word_inputs, pos_inputs, ner_inputs, subj_pos_inputs, obj_pos_inputs):
    ''' Create the embedding layers and return the embeddings, lookup sentence batch and final token dimension. '''
    # create the word embedding layer
    W_emb, W_emb_trainable = _create_embedding(FLAGS.vocab_size, FLAGS.hidden_size, 'word_embedding')
    sent_batch = tf.nn.embedding_lookup(params=W_emb, ids=word_inputs)
    batch_to_concat = [sent_batch]
    dim = FLAGS.hidden_size

    if FLAGS.pos_size > 0:
        POS_emb, _ = _create_embedding(NUM_POS, FLAGS.pos_size, 'pos_embedding')
        pos_batch = tf.nn.embedding_lookup(params=POS_emb, ids=pos_inputs)
        batch_to_concat.append(pos_batch)
        dim += FLAGS.pos_size
    else:
        POS_emb = None
    if FLAGS.ner_size > 0:
        NER_emb, _ = _create_embedding(NUM_NER, FLAGS.ner_size, 'ner_embedding')
        ner_batch = tf.nn.embedding_lookup(params=NER_emb, ids=ner_inputs)
        batch_to_concat.append(ner_batch)
        dim += FLAGS.ner_size
    else:
        NER_emb = None

    # create the position embedding layer
    if FLAGS.pos_emb_size > 0:
        # we need sent_len * 2 embeddings for pos/neg numbers, and another 1 embedding for 0
        P_emb, _ = _create_embedding(FLAGS.sent_len*2+2, FLAGS.pos_emb_size, 'position_embedding')
        subj_pos_batch = tf.nn.embedding_lookup(params=P_emb, ids=subj_pos_inputs)
        obj_pos_batch = tf.nn.embedding_lookup(params=P_emb, ids=obj_pos_inputs)
        # concatenate to form sentence batch
        batch_to_concat += [subj_pos_batch, obj_pos_batch]
        dim += 2*FLAGS.pos_emb_size
    else:
        P_emb = None

    # concatenate to form the final sent_batch
    if len(batch_to_concat) > 1:
        sent_batch = tf.concat(batch_to_concat, 2)
    embeddings = (W_emb, W_emb_trainable, POS_emb, NER_emb, P_emb)

    return embeddings, sent_batch, dim

class SimpleCNNModel():
    """ A 1D CNN model with max-pooling and entity position information. """

    def __init__(self, is_train=True):
        self.is_train = is_train
        #if batch_size is None:
        #    self.batch_size = FLAGS.batch_size # use default batch_size
        #else:
        #    self.batch_size = batch_size
        self.build_graph()

    def build_graph(self):
        # print graph info
        print >> sys.stderr, _get_graph_info()

        self.word_inputs = tf.placeholder(dtype=tf.int32, shape=[None, FLAGS.sent_len], name='input_x')
        self.pos_inputs = tf.placeholder(dtype=tf.int32, shape=[None, FLAGS.sent_len], name='input_pos')
        self.ner_inputs = tf.placeholder(dtype=tf.int32, shape=[None, FLAGS.sent_len], name='input_ner')
        self.labels = tf.placeholder(dtype=tf.int32, shape=[None], name='input_y')
        self.seq_lens = tf.placeholder(dtype=tf.int32, shape=[None], name='input_seq_len')
        # additional inputs: sequence of distance to subj and obj
        self.subj_pos_inputs = tf.placeholder(dtype=tf.int32, shape=[None, FLAGS.sent_len], name='input_subj_pos')
        self.obj_pos_inputs = tf.placeholder(dtype=tf.int32, shape=[None, FLAGS.sent_len], name='input_obj_pos')
        # preprocess the position inputs to convert them into index, by adding (sent_len + 1)
        self.subj_pos_inputs += FLAGS.sent_len + 1
        self.obj_pos_inputs += FLAGS.sent_len + 1
        losses = []

        self.batch_size = tf.shape(self.word_inputs)[0]

        # lookup layer
        with tf.variable_scope('lookup') as scope:
            embeddings, sent_batch, dim = _create_embedding_layers(self.word_inputs, self.pos_inputs, self.ner_inputs, self.subj_pos_inputs, self.obj_pos_inputs)
            self.W_emb, self.W_emb_trainable, self.POS_emb, self.NER_emb, self.P_emb = embeddings
        
        # conv + pooling layer
        with tf.variable_scope('conv') as scope:
            pool_tensors = []
            for window_size in range(FLAGS.min_window, FLAGS.max_window+1):
                filters, wd = _variable_cpu_with_weight_decay(name='filter_'+str(window_size),
                    shape=[window_size, dim, FLAGS.num_filter], initializer=tf.random_normal_initializer(stddev=0.01), wd=FLAGS.l2_reg)
                losses.append(wd)
                conv = tf.nn.conv1d(sent_batch, filters, stride=1, padding='VALID')
                biases = _variable_on_cpu('biases_'+str(window_size), [FLAGS.num_filter], tf.constant_initializer(0.0))
                bias = tf.nn.bias_add(conv, biases)
                relu = tf.nn.relu(bias, name=scope.name)
                # shape of relu: [batch_size, conv_len, num_filter]
                conv_len = relu.get_shape()[1]
                relu = tf.expand_dims(relu, dim=1) # add height to make the relu tensor 4d to support max_pool
                pool = tf.nn.max_pool(relu, ksize=[1,1,conv_len,1], strides=[1,1,1,1], padding='VALID')
                # shape of pool: [batch_size, 1, 1, num_filter]
                pool = tf.squeeze(pool,squeeze_dims=[1,2]) # size: [batch_size, num_filter]
                pool_tensors.append(pool)
            pool_layer = tf.concat(pool_tensors, 1, name='pool')

        # drop out layer
        if self.is_train and FLAGS.dropout > 0:
            pool_dropout = tf.nn.dropout(pool_layer, 1 - FLAGS.dropout)
        else:
            pool_dropout = pool_layer

        # fully-connected layer
        pool_size = (FLAGS.max_window - FLAGS.min_window + 1) * FLAGS.num_filter
        with tf.variable_scope('fc') as scope:
            W, wd = _variable_cpu_with_weight_decay('W', shape=[pool_size, FLAGS.num_class],
                initializer=tf.random_normal_initializer(stddev=0.05), wd=FLAGS.l2_reg)
            losses.append(wd)
            biases = _variable_on_cpu('biases', [FLAGS.num_class], tf.constant_initializer(0.0))
            logits = tf.nn.bias_add(tf.matmul(pool_dropout, W), biases)

        # loss
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=self.labels, name='cross_entropy_per_batch')
        cross_entropy_loss = tf.reduce_mean(cross_entropy, name='cross_entropy_loss')
        losses.append(cross_entropy_loss)
        self.loss = tf.add_n(losses, name='total_loss')
        # self.total_loss = cross_entropy_loss

        # get predictions and probs, shape [batch_size] tensors
        if not self.is_train:
            self.probs = tf.nn.softmax(logits)
            confidence, prediction = tf.nn.top_k(self.probs, k=1)
            self.confidence = tf.squeeze(confidence)
            self.prediction = tf.squeeze(prediction)
        else:
            self.prediction, self.confidence = None, None

        # train on a batch
        self.lr = tf.Variable(1.0, trainable=False)
        if self.is_train:
            opt = _get_optimizer(self.lr)
            grads = opt.compute_gradients(self.loss)
            self.train_op = opt.apply_gradients(grads)
        else:
            self.train_op = tf.no_op()

    def assign_lr(self, session, lr_value):
        session.run(tf.assign(self.lr, lr_value))

    def assign_embedding(self, session, pretrained):
        ''' Note that W_emb_trainable is 1 dimenion less (no pad dimension) than the full W_emb matrix. '''
        session.run(tf.assign(self.W_emb_trainable, pretrained))
