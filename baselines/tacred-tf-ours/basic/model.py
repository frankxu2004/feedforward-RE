import sys
import tensorflow as tf
from tensorflow.contrib import rnn

import data_utils

tf.app.flags.DEFINE_integer('hidden_size', 256, 'Size of word embeddings and hidden layers')
tf.app.flags.DEFINE_integer('num_layers', 2, 'Number of cell layers')
tf.app.flags.DEFINE_integer('pos_size', 0, 'Size of embeddings for POS tags')
tf.app.flags.DEFINE_integer('ner_size', 0, 'Size of embeddings for NER tags')
tf.app.flags.DEFINE_integer('signature_size', 0, 'Size of added signature embeddings into the final representation. 0 = not add.')

tf.app.flags.DEFINE_integer('vocab_size', 35888, 'Vocabulary size')
tf.app.flags.DEFINE_integer('num_class', 42, 'Number of class to consider')
tf.app.flags.DEFINE_integer('sent_len', 96, 'Input sentence length. This is after the padding is performed.')
tf.app.flags.DEFINE_float('max_grad_norm', 5.0, 'The maximum norm used to clip the gradients')
tf.app.flags.DEFINE_float('dropout', 0.5, 'Dropout rate that applies to the LSTM. 0 is no dropout.')

tf.app.flags.DEFINE_boolean('pool', False, 'Add a max pooling layer at the end')

tf.app.flags.DEFINE_boolean('attn', False, 'Whether to use an attention layer')
tf.app.flags.DEFINE_integer('attn_size', 256, 'Size of attention layer')
tf.app.flags.DEFINE_float('attn_stddev', 0.001, 'The attention weights are initialized as normal(0, attn_stddev)')
tf.app.flags.DEFINE_string('attn_query', 'hidden', 'Query vector. hidden: use hidden state; random: random initialization.')

tf.app.flags.DEFINE_string('attn_type', 'non-linear', 'Attention type. non-linear: a perceptron layer; multi: multiplication; dot: dot product between input and query.')
tf.app.flags.DEFINE_integer('attn_pos_emb', 0, 'Attention positional embedding size.')
tf.app.flags.DEFINE_string('attn_pos_type', 'default', 'The way to use attention positional index. default: polarized distances; \
                                                        positive: positive distances; default-normalized: normalized distances.')

tf.app.flags.DEFINE_integer('softmax_hidden', 0, 'The size of hidden layer before the last softmax layer. 0 = no hidden layer.')

tf.app.flags.DEFINE_boolean('bi', False, 'Whether to use a bi-directional lstm')

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

def _get_optimizer(lr):
    if FLAGS.opt == 'adagrad':
        return tf.train.AdagradOptimizer(lr)
    elif FLAGS.opt == 'sgd':
        return tf.train.GradientDescentOptimizer(lr)
    elif FLAGS.opt == 'adam':
        return tf.train.AdamOptimizer(lr)
    else:
        raise Exception("Unsupported optimizer: " + FLAGS.opt)

def _get_lstm_graph_info():
    if FLAGS.bi:
        model_name = 'Bi-LSTM'
    else:
        model_name = 'LSTM'
    info = 'Building %s graph with [%d layers, %d hidden size, %d POS emb, %d NER emb, %d signature emb, %d softmax hidden size]' % \
        (model_name, FLAGS.num_layers, FLAGS.hidden_size, FLAGS.pos_size, FLAGS.ner_size, FLAGS.signature_size, FLAGS.softmax_hidden)
    if FLAGS.pool:
        info += ' and a max-pooling layer'
    if FLAGS.attn:
        info += ' and an attention layer [%d attn size, query=%s, type=%s, pos_emb=%d, pos_type=%s]' % \
                (FLAGS.attn_size, FLAGS.attn_query, FLAGS.attn_type, FLAGS.attn_pos_emb, FLAGS.attn_pos_type)
    info += ' ...'
    return info

def _get_rnn_cell(hidden_size, num_layers, is_train, dropout):
    '''
        Return a LSTM cell. Useful to get it modularized.
    '''
    cell = rnn.BasicLSTMCell(num_units=hidden_size)
    if is_train:
        cell = rnn.DropoutWrapper(cell, output_keep_prob=1-dropout)
    cell = rnn.MultiRNNCell([cell] * num_layers)
    return cell

def max_over_time(inputs, index, seq_lens):
    ''' Use with tf.map_fn()
        Args:
            inputs: [batch_size, sent_len, dim]
            seq_lens: [batch_size]
        Return:
            output: [dim]
    '''
    l = seq_lens[index]
    valid_inputs = inputs[index, :l] # [l, dim]
    output = tf.reduce_max(valid_inputs, reduction_indices=[0])
    return output

def attention_over_time(inputs, query, pos_vectors, index, seq_lens, dim):
    ''' Use with tf.map_fn()
        Args:
            inputs: [batch_size, sent_len, dim]
            query: [batch_size, dim]
            pos_vectors: a pair of [batch_size, attn_pos_emb]
            seq_lens: [batch_size]
        Return:
            attention: attention distribution (after softmax) with shape [batch_size, sent_len]
    '''
    l = seq_lens[index]
    valid_inputs = inputs[index, :l] # [h_0, .., h_i, ...], shape [l, dim]
    current_query = tf.expand_dims(query[index, :], 0) # q, shape [1, dim]
    assert(len(pos_vectors) == 2)
        
    # initialization: weights ~ n(0, stddev), biases = 0, V = zero vector
    attn_bias_init = 0.0 # the original attention paper uses 0.0001 for weights initialization
    attn_size = FLAGS.attn_size

    if FLAGS.attn_type == 'non-linear':
        # non-linear attention: e = V' \dot tanh(Wh + Wq)
        attn_W_h = tf.get_variable('attn_W_h', shape=[attn_size, dim], initializer=tf.random_normal_initializer(stddev=FLAGS.attn_stddev))
        attn_W_q = tf.get_variable('attn_W_q', shape=[attn_size, dim], initializer=tf.random_normal_initializer(stddev=FLAGS.attn_stddev))
        attn_b = tf.get_variable('attn_b', shape=[attn_size], initializer=tf.constant_initializer(attn_bias_init))
        attn_V = tf.get_variable('attn_V', shape=[attn_size, 1], initializer=tf.constant_initializer(0.0))
        attn_V_b = tf.get_variable('attn_V_b', shape=[1], initializer=tf.constant_initializer(attn_bias_init))
        
        x1 = tf.matmul(valid_inputs, tf.transpose(attn_W_h)) # input transformation [l, attn]
        x2 = tf.matmul(current_query, tf.transpose(attn_W_q)) # query transformation [1, attn]

        if FLAGS.attn_pos_emb == 0:
            y = tf.tanh(tf.nn.bias_add(x1 + x2, attn_b)) # [l, attn_size] # non-linear
        elif FLAGS.attn_pos_emb > 0: # otherwise the pos_vectors will be a pair of None
            subj_pos_vectors = pos_vectors[0][index, :l]
            obj_pos_vectors = pos_vectors[1][index, :l]
            attn_W_subj = tf.get_variable('attn_W_subj', shape=[attn_size, FLAGS.attn_pos_emb], initializer=tf.random_normal_initializer(stddev=FLAGS.attn_stddev))
            attn_W_obj = tf.get_variable('attn_W_obj', shape=[attn_size, FLAGS.attn_pos_emb], initializer=tf.random_normal_initializer(stddev=FLAGS.attn_stddev))
            x3 = tf.matmul(subj_pos_vectors, tf.transpose(attn_W_subj)) # subj position transformation
            x4 = tf.matmul(obj_pos_vectors, tf.transpose(attn_W_obj)) # obj position transformation
            y = tf.tanh(tf.nn.bias_add(x1 + x2 + x3 + x4, attn_b))
            
        e = tf.transpose(tf.nn.bias_add(tf.matmul(y, attn_V), attn_V_b)) # [1, l]

    elif FLAGS.attn_type == 'dot':
        # dot attention: e = h \dot q
        attn_b = tf.get_variable('attn_b', shape=[1], initializer=tf.constant_initializer(attn_bias_init))
        e = tf.nn.bias_add(tf.matmul(valid_inputs, tf.transpose(current_query)), attn_b) # [l, 1]
        e = tf.transpose(e) # [1, l]
    elif FLAGS.attn_type == 'multi':
        # multiplication attention: e = h \dot W \dot q
        attn_W = tf.get_variable('attn_W', shape=[dim, dim], initializer=tf.random_normal_initializer(stddev=FLAGS.attn_stddev))
        attn_b = tf.get_variable('attn_b', shape=[1], initializer=tf.constant_initializer(attn_bias_init))
        Wq = tf.matmul(attn_W, tf.transpose(current_query))
        e = tf.nn.bias_add(tf.matmul(valid_inputs, Wq), attn_b) # [l, 1]
        e = tf.transpose(e) # [1, l]
    else:
        raise Exception('Unsupported attention type: ' + FLAGS.attn_type)

    attention = tf.nn.softmax(e) # [1, l]
    attention = tf.pad(attention, paddings=[[0,0],[0,FLAGS.sent_len-l]], mode='CONSTANT') # pad to [1, sent_len]
    return attention

def _create_attention_layer(rnn_outputs, rnn_final_hidden, pos_vectors, seq_lens, dim, batch_size):
        
    # generating query vectors, with shape [batch_size, dim]
    if FLAGS.attn_query == 'random':
        attn_query = tf.get_variable('attn_query', shape=[1, dim], initializer=tf.random_normal_initializer(stddev=0.01)) # 0.01 is best based on fine-tuning
        attn_query = tf.tile(attn_query, multiples=[batch_size, 1]) # replicate attn_query to form shape [batch_size, dim]
    elif FLAGS.attn_query == 'hidden':
        attn_query = rnn_final_hidden
    else:
        raise Exception('Unsupported attention query type: ' + FLAGS.attn_query)

    attention = tf.map_fn(lambda idx: attention_over_time(rnn_outputs, attn_query, pos_vectors, idx, seq_lens, dim), tf.range(0, batch_size), dtype=tf.float32)
    attn_final_state = tf.reduce_sum(tf.reshape(attention, [-1, FLAGS.sent_len, 1]) * rnn_outputs, [1]) # shape: [batch_size, dim]
    return attention, attn_final_state

def _create_embedding_layers(word_inputs, pos_inputs, ner_inputs, signature_inputs, subj_pos_inputs, obj_pos_inputs, num_pos_emb):
    ''' Create the embedding layers and return the embeddings, lookup sentence batch. '''
    # create the word embedding layer
    W_emb = _variable_on_cpu('embedding', shape=[FLAGS.vocab_size, FLAGS.hidden_size],
        initializer=tf.random_uniform_initializer(minval=-1.0, maxval=1.0))
    sent_batch = tf.nn.embedding_lookup(params=W_emb, ids=word_inputs) # [batch_size, sent_len, dim]
    batch_to_concat = [sent_batch]
    
    # create embeddings for POS and NERs
    if FLAGS.pos_size > 0:
        POS_emb = _variable_on_cpu(name='pos_embedding', shape=[NUM_POS, FLAGS.pos_size],
                initializer=tf.random_uniform_initializer(minval=-1.0, maxval=1.0), trainable=True)
        pos_batch = tf.nn.embedding_lookup(params=POS_emb, ids=pos_inputs)
        batch_to_concat.append(pos_batch)
    else:
        POS_emb = None
    if FLAGS.ner_size > 0:
        NER_emb = _variable_on_cpu(name='ner_embedding', shape=[NUM_NER, FLAGS.ner_size],
                initializer=tf.random_uniform_initializer(minval=-1.0, maxval=1.0), trainable=True)
        ner_batch = tf.nn.embedding_lookup(params=NER_emb, ids=ner_inputs)
        batch_to_concat.append(ner_batch)
    else:
        NER_emb = None

    # create the position embedding layer used in the attention
    if FLAGS.attn and FLAGS.attn_pos_emb > 0:
        # Unlike CNN, do not need to create untrainable zero vector for padding, because padding will be ignored anyway.
        # we need sent_len * 2 embeddings for pos/neg numbers, and another 1 embedding for 0 (padding is 0 too)
        P_emb = _variable_on_cpu(name='position_embedding', shape=[num_pos_emb, FLAGS.attn_pos_emb],
                    initializer=tf.random_uniform_initializer(minval=-1.0, maxval=1.0), trainable=True)
        subj_pos_batch = tf.nn.embedding_lookup(params=P_emb, ids=subj_pos_inputs)
        obj_pos_batch = tf.nn.embedding_lookup(params=P_emb, ids=obj_pos_inputs)
    else:
        P_emb, subj_pos_batch, obj_pos_batch = None, None, None

    # create signature embeddings
    if FLAGS.signature_size > 0:
        SUBJ_emb = _variable_on_cpu(name='subj_embedding', shape=[NUM_SUBJ_NER, FLAGS.signature_size],
                initializer=tf.random_uniform_initializer(minval=-1.0, maxval=1.0), trainable=True)
        OBJ_emb = _variable_on_cpu(name='obj_embedding', shape=[NUM_OBJ_NER, FLAGS.signature_size],
                initializer=tf.random_uniform_initializer(minval=-1.0, maxval=1.0), trainable=True)
        subj_signature_inputs, obj_signature_inputs = tf.unpack(signature_inputs, num=2, axis=1)
        subj_signature_batch = tf.nn.embedding_lookup(params=SUBJ_emb, ids=subj_signature_inputs)
        obj_signature_batch = tf.nn.embedding_lookup(params=OBJ_emb, ids=obj_signature_inputs)
        signature_batch = tf.concat(values=[subj_signature_batch, obj_signature_batch], axis=1)
    else:
        SUBJ_emb, OBJ_emb, signature_batch = None, None, None
    
    # concatenate to form the final sent_batch
    if len(batch_to_concat) > 1:
        sent_batch = tf.concat(batch_to_concat, 2)

    embeddings = (W_emb, P_emb, POS_emb, NER_emb, SUBJ_emb, OBJ_emb)
    return embeddings, sent_batch, subj_pos_batch, obj_pos_batch, signature_batch

def normalize_and_ceil_position_tensor(input_tensor, positive_output=True):
    ''' Do floor for negative, and ceil for positive. 
        Args: all_positive: whether the ouput are all positive (so that we only do positive ceiling).'''
    if positive_output:
        normalized_tensor = tf.ceil(tf.cast(tf.abs(input_tensor), tf.float32) / float(FLAGS.sent_len) * 10.0)
        return tf.cast(normalized_tensor, tf.int32)
    else:
        signs = tf.sign(input_tensor) # keep signs
        normalized_tensor = tf.ceil(tf.cast(tf.abs(input_tensor), tf.float32) / float(FLAGS.sent_len) * 10.0)
        return tf.mul(signs, tf.cast(normalized_tensor, tf.int32))

def _preprocess_position_inputs(subj_pos_inputs, obj_pos_inputs):
    ''' Preprocess the position inputs so that we can have different types. Also need to return corresponding position embedding number.
        To use this function, the padding index needs to be zero. '''
    if FLAGS.attn_pos_emb <= 0: # no need to process
        return subj_pos_inputs, obj_pos_inputs, 0
    if FLAGS.attn_pos_type == 'default':
        return subj_pos_inputs + FLAGS.sent_len, obj_pos_inputs + FLAGS.sent_len, FLAGS.sent_len*2 + 1
    elif FLAGS.attn_pos_type == 'positive': # use all positive indices
        return tf.abs(subj_pos_inputs), tf.abs(obj_pos_inputs), FLAGS.sent_len + 1
    elif FLAGS.attn_pos_type == 'default-normalized': # normalize the distance to 20 categories
        return normalize_and_ceil_position_tensor(subj_pos_inputs, False) + 10, normalize_and_ceil_position_tensor(obj_pos_inputs, False) + 10, 21
    elif FLAGS.attn_pos_type == 'positive-normalized':
        return normalize_and_ceil_position_tensor(subj_pos_inputs, True), normalize_and_ceil_position_tensor(obj_pos_inputs, True), 11
    else:
        raise Exception('Attention position input type not supported: %s' % FLAGS.attn_pos_type)

class LSTMModel(object):
    """
    A LSTM model that reads sentence, and predicts a label at the end. 
    An optional max-pooling layer or attention layer can be appended afterwards.
    An optinal hidden layer can be added before the final softmax layer.
    """

    def __init__(self, is_train=True):
        self.is_train = is_train
        self.build_graph()

    def build_graph(self):
        # sanity check
        if FLAGS.pool and FLAGS.attn:
            raise Exception("Max-pooling layer and attention layer cannot be added at the same time.")
        if FLAGS.attn and FLAGS.attn_pos_emb > 0 and FLAGS.attn_type != 'non-linear':
            raise Exception("Attention type %s does not support positional vector transformation." % FLAGS.attn_type)

        # print graph info
        print >> sys.stderr, _get_lstm_graph_info()

        self.word_inputs = tf.placeholder(dtype=tf.int32, shape=[None, FLAGS.sent_len], name='input_x')
        self.pos_inputs = tf.placeholder(dtype=tf.int32, shape=[None, FLAGS.sent_len], name='input_pos')
        self.ner_inputs = tf.placeholder(dtype=tf.int32, shape=[None, FLAGS.sent_len], name='input_ner')
        self.signature_inputs = tf.placeholder(dtype=tf.int32, shape=[None, 2], name='input_signature')
        self.labels = tf.placeholder(dtype=tf.int32, shape=[None], name='input_y')
        self.seq_lens = tf.placeholder(dtype=tf.int32, shape=[None], name='input_seq_len')
        
        # for positional embeddings in attention calculation
        # additional inputs: sequence of distance to subj and obj
        self.subj_pos_inputs = tf.placeholder(dtype=tf.int32, shape=[None, FLAGS.sent_len], name='input_subj_pos')
        self.obj_pos_inputs = tf.placeholder(dtype=tf.int32, shape=[None, FLAGS.sent_len], name='input_obj_pos')
        
        # preprocess the position inputs to convert them into index, by adding (sent_len + 1)
        self.subj_pos_indices, self.obj_pos_indices, num_pos_emb = _preprocess_position_inputs(self.subj_pos_inputs, self.obj_pos_inputs)

        self.batch_size = tf.shape(self.word_inputs)[0]

        # rnn cell
        dim = FLAGS.hidden_size + FLAGS.pos_size + FLAGS.ner_size
        if FLAGS.bi:
            cell_fw, cell_bw = _get_rnn_cell(dim, FLAGS.num_layers, self.is_train, FLAGS.dropout), \
                    _get_rnn_cell(dim, FLAGS.num_layers, self.is_train, FLAGS.dropout)
        else:
            cell = _get_rnn_cell(dim, FLAGS.num_layers, self.is_train, FLAGS.dropout)

        # embedding layer
        embeddings, sent_batch, subj_pos_batch, obj_pos_batch, signature_batch = \
                _create_embedding_layers(self.word_inputs, self.pos_inputs, self.ner_inputs, self.signature_inputs, self.subj_pos_indices, self.obj_pos_indices, num_pos_emb)
        self.W_emb, self.P_emb, self.POS_emb, self.NER_emb, self.SUBJ_emb, self.OBJ_emb = embeddings # unpack embeddings
        
        if self.is_train:
            sent_batch = tf.nn.dropout(sent_batch, 1-FLAGS.dropout)

        # rnn layer
        # NOTE that rnn is doing dynamic batching now so outputs will be tailed by zero vectors in a batch
        if FLAGS.bi:
            rnn_outputs, rnn_final_states = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, sent_batch, dtype=tf.float32, sequence_length=self.seq_lens)
            rnn_outputs = tf.concat(rnn_outputs, 2) # the original is a tuple, we need to concatenate them
            rnn_final_states_fw, rnn_final_states_bw = rnn_final_states
            rnn_final_state = tf.concat([rnn_final_states_fw[-1][1], rnn_final_states_bw[-1][1]], 1)
            dim = 2*dim
        else:
            rnn_outputs, rnn_final_states = tf.nn.dynamic_rnn(cell, sent_batch, dtype=tf.float32, sequence_length=self.seq_lens)
            # rnn_final_states is tuple of tuple for LSTM, outputs is a tensor [batch_size, sent_len, dim]
            rnn_final_state = rnn_final_states[-1][1]

        # [optional] add pooling or attention layer
        if FLAGS.pool:
            final_hidden = tf.map_fn(lambda idx: max_over_time(rnn_outputs, idx, self.seq_lens), tf.range(0, self.batch_size), dtype=tf.float32)
        elif FLAGS.attn:
            # attention layer
            self.attention, attn_final_state = _create_attention_layer(rnn_outputs, rnn_final_state, [subj_pos_batch, obj_pos_batch], \
                 self.seq_lens, dim, self.batch_size)
            final_hidden = attn_final_state
        else:
            final_hidden = rnn_final_state

        # [optional] add a hidden layer before softmax
        softmax_hidden_dim = FLAGS.softmax_hidden
        if softmax_hidden_dim != 0:
            self.softmax_hidden_w = tf.get_variable('softmax_hidden_w', shape=[dim, softmax_hidden_dim])
            self.softmax_hidden_b = tf.get_variable('softmax_hidden_b', shape=[softmax_hidden_dim])
            final_hidden = tf.tanh(tf.nn.bias_add(tf.matmul(final_hidden, self.softmax_hidden_w), self.softmax_hidden_b))
            if self.is_train: # apply dropout after this layer
                final_hidden = tf.nn.dropout(final_hidden, 1-FLAGS.dropout)
        else:
            # softmax_hidden == 0 then skip this layer
            softmax_hidden_dim = dim

        # add signature embeddings into final representation if specified
        if FLAGS.signature_size > 0:
            final_hidden = tf.concat(values=[final_hidden, signature_batch], axis=1)
            softmax_hidden_dim += 2 * FLAGS.signature_size

        # softmax layer
        self.softmax_w = tf.get_variable('softmax_w', shape=[softmax_hidden_dim, FLAGS.num_class])
        self.softmax_b = tf.get_variable('softmax_b', shape=[FLAGS.num_class])
        self.logits = tf.nn.bias_add(tf.matmul(final_hidden, self.softmax_w), self.softmax_b)

        # loss and accuracy
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=self.labels, name='cross_entropy_per_batch')
        self.loss = tf.reduce_mean(cross_entropy, name='cross_entropy_loss')

        correct_prediction = tf.to_int32(tf.nn.in_top_k(self.logits, self.labels, 1))
        self.true_count_op = tf.reduce_sum(correct_prediction)

        # get predictions and probs, shape [batch_size] tensors
        self.probs = tf.nn.softmax(self.logits)
        self.confidence, self.prediction = tf.nn.top_k(self.probs, k=1)
        self.confidence = tf.squeeze(self.confidence)
        self.prediction = tf.squeeze(self.prediction)
        
        # train on a batch
        self.lr = tf.Variable(1.0, trainable=False)
        if self.is_train:
            opt = _get_optimizer(self.lr)
            tvars = tf.trainable_variables()
            grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, tvars), FLAGS.max_grad_norm)
            self.train_op = opt.apply_gradients(zip(grads, tvars))
        else:
            self.train_op = tf.no_op()
        return

    def assign_lr(self, session, lr_value):
        session.run(tf.assign(self.lr, lr_value))

    def assign_embedding(self, session, pretrained):
        session.run(tf.assign(self.W_emb, pretrained))
