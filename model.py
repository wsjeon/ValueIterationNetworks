import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim import conv2d
from tensorflow.contrib.slim import fully_connected as fc
from utils import conv2d_flipkernel
from data  import process_gridworld_data
from utils import fmt_row
import time

class Agent(object):
  def build_model(self):
    FLAGS = self.FLAGS
    FLAGS = tf.app.flags.FLAGS
    self.X  = tf.placeholder(tf.float32, [None, FLAGS.imsize, FLAGS.imsize, FLAGS.ch_i])
    self.S1 = tf.placeholder(tf.int32, [None, FLAGS.statebatchsize])
    self.S2 = tf.placeholder(tf.int32, [None, FLAGS.statebatchsize])

    k    = FLAGS.k    # Number of value iterations performed
    ch_i = FLAGS.ch_i # Channels in input layer
    ch_h = FLAGS.ch_h # Channels in initial hidden layer
    ch_q = FLAGS.ch_q # Channels in q layer (~actions)
    state_batch_size = FLAGS.statebatchsize # k+1 state inputs for each channel

    tf.set_random_seed(0)

    init = tf.random_normal_initializer(stddev = 0.01)

    with slim.arg_scope([conv2d, fc], activation_fn = None, weights_initializer = init,
        biases_initializer = None):
      h = conv2d(self.X, ch_h, [3, 3], biases_initializer = init, scope = "conv1")
      r = conv2d(h, 1, [3, 3], scope = "conv2")
      v = tf.zeros(tf.shape(r))
      for i in range(k + 1):
        q = conv2d(tf.concat([r, v], axis = 3), ch_q, [3, 3], reuse = True, scope = "conv3")
        v = tf.reduce_max(q, axis = 3, keep_dims = True)

      q = tf.transpose(q, perm=[0, 3, 1, 2])
      bs = tf.shape(q)[0]
      rprn = tf.reshape(tf.tile(tf.reshape(tf.range(bs), [-1, 1]), [1, state_batch_size]), [-1])
      ins1 = tf.cast(tf.reshape(self.S1, [-1]), tf.int32)
      ins2 = tf.cast(tf.reshape(self.S2, [-1]), tf.int32)
      idx_in = tf.transpose(tf.stack([ins1, ins2, rprn]), [1, 0])
      q_out = tf.gather_nd(tf.transpose(q, [2, 3, 0, 1]), idx_in)

      # add logits
      self.logits = fc(q_out, 8, scope = "fc0")
      # softmax output weights
      self.net = tf.nn.softmax(self.logits)

  def build_loss(self):
    self.y = tf.placeholder(tf.int32, [None])
    y_ = tf.cast(self.y, tf.int64)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=self.logits, labels=y_)
    self.loss = tf.reduce_mean(cross_entropy)

  def build_optimizer(self):
    optimizer = tf.train.RMSPropOptimizer(learning_rate=self.FLAGS.lr, epsilon=1e-6, centered=True)
    self.optimizer = optimizer.minimize(self.loss)

  def build_summary(self):
    self.cp = tf.cast(tf.argmax(self.net, axis = 1), tf.int32)
    self.err = tf.reduce_mean(tf.cast(tf.not_equal(self.cp, self.y), dtype=tf.float32))
    for var in tf.trainable_variables():
      tf.summary.histogram(var.op.name, var)
    self.summary_op = tf.summary.merge_all()
    self.summary_writer = tf.summary.FileWriter(self.FLAGS.logdir)

  def __init__(self):
    self.FLAGS = tf.app.flags.FLAGS
    FLAGS = self.FLAGS

    np.random.seed(self.FLAGS.seed)

    """ TensorFlow graph construction """
    self.build_model()
    self.build_loss()
    self.build_optimizer()
    self.build_summary()

    """ Open TensorFlow session and initialize variables. """
    self.sess = tf.Session()
    self.sess.run(tf.global_variables_initializer())

    Xtrain, S1train, S2train, ytrain, Xtest, S1test, S2test, ytest = process_gridworld_data(input=FLAGS.input, imsize=FLAGS.imsize)

    batch_size = FLAGS.batchsize
    print(fmt_row(10, ["Epoch", "Train Cost", "Train Err", "Epoch Time"]))
    for epoch in range(int(FLAGS.epochs)):
        tstart = time.time()
        avg_err, avg_cost = 0.0, 0.0
        num_batches = int(Xtrain.shape[0]/batch_size)
        # Loop over all batches
        for i in range(0, Xtrain.shape[0], batch_size):
            j = i + batch_size
            if j <= Xtrain.shape[0]:
                # Run optimization op (backprop) and cost op (to get loss value)
                fd = {self.X: Xtrain[i:j],
                      self.S1: S1train[i:j],
                      self.S2: S2train[i:j],
                      self.y: ytrain[i * FLAGS.statebatchsize:j * FLAGS.statebatchsize]}
                _, e_, c_ = self.sess.run([self.optimizer, self.err, self.loss], feed_dict=fd)
                avg_err += e_
                avg_cost += c_
        # Display logs per epoch step
        if epoch % FLAGS.display_step == 0:
            elapsed = time.time() - tstart
            print(fmt_row(10, [epoch, avg_cost/num_batches, avg_err/num_batches, elapsed]))
        summary = tf.Summary()
        summary.ParseFromString(self.sess.run(self.summary_op))
        summary.value.add(tag='Average error', simple_value=float(avg_err/num_batches))
        summary.value.add(tag='Average cost', simple_value=float(avg_cost/num_batches))
        self.summary_writer.add_summary(summary, epoch)
    print("Finished training!")

    # Test model
    correct_prediction = tf.cast(tf.argmax(self.net, 1), tf.int32)
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(tf.not_equal(correct_prediction, self.y), dtype=tf.float32))
    acc = accuracy.eval({self.X: Xtest, self.S1: S1test, self.S2: S2test, self.y: ytest})
    print "Accuracy: {}%".format(100 * (1 - acc))

    self.sess.close()
