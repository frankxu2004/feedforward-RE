import sys
import tensorflow as tf

# cnn parameters
tf.app.flags.DEFINE_integer('hidden_size', 256, 'Size of word embeddings and hidden layers')
tf.app.flags.DEFINE_string('cnn_struct', '3-64,3-128,3-256,2-p,3-128', 'The CNN structure to use.')
tf.app.flags.DEFINE_integer('vocab_size', 35888, 'Vocabulary size')
tf.app.flags.DEFINE_integer('num_class', 42, 'Number of class to consider')
tf.app.flags.DEFINE_integer('sent_len', 96, 'Input sentence length. This is after the padding is performed.')
tf.app.flags.DEFINE_float('conv_dropout', 0, 'Dropout rate that applies between the CNN and RNN layer.')
tf.app.flags.DEFINE_float('rnn_dropout', 0.5, 'Dropout rate that applies to the RNN. 0 is no dropout.')
tf.app.flags.DEFINE_float('l2_reg', 0.001, 'l2 regularization weight')

# rnn layer params
tf.app.flags.DEFINE_integer('num_layers', 1, 'Number of cell layers')
tf.app.flags.DEFINE_float('max_grad_norm', 5.0, 'The maximum norm used to clip the gradients')
tf.app.flags.DEFINE_boolean('pool', False, 'Add a max pooling layer at the end of RNN')

tf.app.flags.DEFINE_string('opt', 'adagrad', 'The optimizer to use, must be one of adagrad, sgd, adam.')

FLAGS = tf.app.flags.FLAGS

def _variable_on_cpu(name, shape, initializer, trainable=True):
    with tf.device('/cpu:0'):
        var = tf.get_variable(name, shape, initializer=initializer, trainable=trainable)
    return var

def _variable_cpu_with_weight_decay(name, shape, initializer, wd):
    var = _variable_on_cpu(name, shape, initializer)
    if wd is not None and wd != 0.:
        weight_decay = tf.mul(tf.nn.l2_loss(var), wd, name='weight_loss')
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

def _get_cnn_struct_from_str(struct):
    """
    Parse a string representation of a CNN structure. In the string, layers are separated by comma.
    For a convolution layer, "3-64" means a width 3 convolution layer with 64 feature maps. A max-pooling layer
    is represented by "2-p", which means a max-pooling layer of width 2.
    Thus, "3-64,3-128,2-p,1-32" means a 4-layer network, with 2 convolution layers followed by a max-pooling layer then
    a final convolution layer.
    Note that padding will always be SAME, and stride will always be 1 for conv layers.
    Return value is a list of layers. Each layer is represented as a dict with key: type, width, num_filter.
    num_filter will only be present if type == 'conv'.
    """
    def parse_error(str):
        raise Exception("Cannot parse CNN structure representation: " + str)
    layers = []
    array = struct.strip().split(',')
    if len(array) == 0:
        parse_error(struct)
    for s in array:
        layer = {}
        layer_info = s.split('-')
        if len(layer_info) != 2:
            parse_error(s)
        if layer_info[1].lower() == 'p':
            layer['type'] = 'pool'
        else:
            layer['type'] = 'conv'
            layer['num_filter'] = int(layer_info[1])
        layer['width'] = int(layer_info[0])
        layers.append(layer)
    return layers

def _get_lstm_graph_info(layers):
    info = 'Building Conv-RNN model with CNN structure '
    cnn_info = '['
    for l in layers:
        cnn_info += l['type']
        if 'num_filter' in l:
            cnn_info += str(l['num_filter'])
        cnn_info += ','
    cnn_info = cnn_info[:-1] + ']'
    info += cnn_info + ' and LSTM with [%d layers]' % FLAGS.num_layers
    if FLAGS.pool:
        info += ' and a max-pooling layer'
    info += '\n- Other model params:\n'
    info += '\tconv_dropout: %g\n' % FLAGS.conv_dropout
    info += '\trnn_dropout: %g\n' % FLAGS.rnn_dropout
    info += '\tl2_reg: %g' % FLAGS.l2_reg
    return info

class ConvLSTMModel():
    """ A Conv-LSTM model that stacks CNN and RNN together. """

    def __init__(self, is_train=True, batch_size=None):
        self.is_train = is_train
        if batch_size is None:
            self.batch_size = FLAGS.batch_size # use default batch_size
        else:
            self.batch_size = batch_size
        self.build_graph()

    def build_graph(self):
        # Get graph info from parameters
        layers = _get_cnn_struct_from_str(FLAGS.cnn_struct)
        print >> sys.stderr, _get_lstm_graph_info(layers)

        self.word_inputs = tf.placeholder(dtype=tf.int32, shape=[None, FLAGS.sent_len], name='input_x')
        self.labels = tf.placeholder(dtype=tf.int32, shape=[None], name='input_y')
        self.seq_lens = tf.placeholder(dtype=tf.int32, shape=[None], name='input_seq_len')
        losses = []

        # lookup layer
        with tf.variable_scope('lookup') as scope:
            # create a non-trainable pad vector
            pad_emb = _variable_on_cpu(name='pad_emb', shape=[1,FLAGS.hidden_size],
                initializer=tf.constant_initializer(0.0), trainable=False)
            word_emb = _variable_on_cpu(name='word_emb', shape=[FLAGS.vocab_size-1, FLAGS.hidden_size], 
                initializer=tf.random_uniform_initializer(minval=-1.0, maxval=1.0), trainable=True)
            self.W_emb = tf.concat(0, [pad_emb, word_emb], name='embedding')
            # sent_batch is of shape: (batch_size, sent_len, emb_size)
            sent_batch = tf.nn.embedding_lookup(params=self.W_emb, ids=self.word_inputs)

        # cnn input_tensor shape is always: [batch, in_width, in_channels], here in_width = sent_len, in_channels = hidden_size
        input_tensor = sent_batch
        # build CNN graph based on parsed result
        for i,l in enumerate(layers):
            layer_type = l['type']
            window_size = l['width']
            in_channels = input_tensor.get_shape()[2]
            print >> sys.stderr, "- CNN layer %d (%s) shape:" % (i+1, layer_type)
            print >> sys.stderr, "\tinput: " + str(input_tensor.get_shape())
            with tf.variable_scope(layer_type + str(i+1)) as scope:
                if layer_type == 'conv':
                    out_channels = l['num_filter']
                    filters, wd = _variable_cpu_with_weight_decay(name='filters', shape=[window_size, in_channels, out_channels], 
                        initializer=tf.random_normal_initializer(stddev=0.05), wd=FLAGS.l2_reg)
                    losses.append(wd)
                    conv = tf.nn.conv1d(input_tensor, filters, stride=1, padding='SAME')
                    biases = _variable_on_cpu('biases', [out_channels], tf.constant_initializer(0.0))
                    relu = tf.nn.relu(tf.nn.bias_add(conv, biases), name=scope.name)
                    input_tensor = relu # reset input_tensor for the next layer
                elif layer_type == 'pool':
                    input_tensor = tf.expand_dims(input_tensor, dim=1) # add height to make the input_tensor 4d to support max_pool
                    pool = tf.nn.max_pool(input_tensor, ksize=[1,1,window_size,1], strides=[1,1,window_size,1], padding='SAME')
                    input_tensor = tf.squeeze(pool, squeeze_dims=[1]) # we squeeze the tensor back
                else:
                    raise Exception("Unrecognized layer type at cnn layer %d: %s" % (i+1, layer_type))
            print >> sys.stderr, "\toutput: " + str(input_tensor.get_shape()) # input_tensor is now actually output tensor
        cnn_output = input_tensor # we always reset input_tensor to be output in the layers, so we point output to that

        # dropout layer between cnn and rnn
        if self.is_train and FLAGS.conv_dropout > 0:
            cnn_dropout = tf.nn.dropout(cnn_output, 1 - FLAGS.conv_dropout)
        else:
            cnn_dropout = cnn_output

        # rnn layer
        dim = int(cnn_dropout.get_shape()[-1]) # dim = out_channels of the last cnn layer
        with tf.variable_scope('rnn') as scope:
            cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=dim)
            if self.is_train and FLAGS.rnn_dropout > 0:
                cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=1-FLAGS.rnn_dropout)
            cell = tf.nn.rnn_cell.MultiRNNCell([cell] * FLAGS.num_layers)

            # input: cnn_dropout, size [batch_size, conv_len, dim]
            rnn_outputs, rnn_final_states = tf.nn.dynamic_rnn(cell, cnn_dropout, dtype=tf.float32) # final_state is of shape [batch_size, *], outputs is list of [batch_size, dim]
            rnn_final_state = rnn_final_states[-1][1]

            if FLAGS.pool:
                final_hidden = tf.reduce_max(rnn_outputs, reduction_indices=[1]) # do a max-pooling
            else:
                final_hidden = rnn_final_state

        # softmax layer
        self.softmax_w = tf.get_variable('softmax_w', shape=[dim, FLAGS.num_class])
        self.softmax_b = tf.get_variable('softmax_b', shape=[FLAGS.num_class])
        self.logits = tf.nn.bias_add(tf.matmul(final_hidden, self.softmax_w), self.softmax_b)

        # loss
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(self.logits, self.labels, name='cross_entropy_per_batch')
        cross_entropy_loss = tf.reduce_mean(cross_entropy, name='cross_entropy_loss')
        losses.append(cross_entropy_loss)
        self.loss = tf.add_n(losses, name='total_loss')
        # self.total_loss = cross_entropy_loss

        # get predictions and probs, shape [batch_size] tensors
        if not self.is_train:
            self.probs = tf.nn.softmax(self.logits)
            confidence, prediction = tf.nn.top_k(self.probs, k=1)
            self.confidence = tf.squeeze(confidence)
            self.prediction = tf.squeeze(prediction)
        else:
            self.prediction, self.confidence = None, None

        # train on a batch
        self.lr = tf.Variable(1.0, trainable=False)
        if self.is_train:
            opt = _get_optimizer(self.lr)
            tvars = tf.trainable_variables()
            grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, tvars), FLAGS.max_grad_norm)
            self.train_op = opt.apply_gradients(zip(grads, tvars))
        else:
            self.train_op = tf.no_op()

    def assign_lr(self, session, lr_value):
        session.run(tf.assign(self.lr, lr_value))

    def assign_embedding(self, session, pretrained):
        session.run(tf.assign(self.W_emb, pretrained))

def main():
    layers = _get_cnn_struct_from_str('3-64,3-128,3-256,2-p,1-32,3-128')
    print >> sys.stderr, layers

if __name__ == '__main__':
    main()
