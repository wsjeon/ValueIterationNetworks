import tensorflow as tf
import numpy as np
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim import fully_connected as fc
from tensorflow.contrib.slim import conv2d
from utils import conv2d_flipkernel
from data import process_gridworld_data
from utils import fmt_row
import time

FLAGS = tf.app.flags.FLAGS

class Environment(object):
  def __init__(self):
    self.Xtrain, self.S1train, self.S2train, self.ytrain, self.Xtest, self.S1test, self.S2test, self.ytest =\
        process_gridworld_data(input=FLAGS.input, imsize=FLAGS.MAZE_SIZE)

class Agent(object):
  def __init__(self, env):
    self.env = env
    self.build_net()

    self.build_loss()
    self.build_optimizer()
    self.build_summary()
    self.sess = tf.Session()
    self.sess.run(tf.global_variables_initializer())

  def build_net(self):
    self.X  = tf.placeholder(tf.float32, [None, FLAGS.MAZE_SIZE, FLAGS.MAZE_SIZE, 2])
    # symbolic input batches of vertical positions
    self.S1 = tf.placeholder(tf.int32, [None, FLAGS.statebatchsize])
    # symbolic input batches of horizontal positions
    self.S2 = tf.placeholder(tf.int32, [None, FLAGS.statebatchsize])
    self.y  = tf.placeholder(tf.int32, [None])


    k    = FLAGS.k    # Number of value iterations performed
    state_batch_size = FLAGS.statebatchsize # k+1 state inputs for each channel

    bias  = tf.Variable(np.random.randn(1, 1, 1, 150)    * 0.01, dtype=tf.float32)
    # weights from inputs to q layer (~reward in Bellman equation)
    w0    = tf.Variable(np.random.randn(3, 3, 2, 150) * 0.01, dtype=tf.float32)
    w1    = tf.Variable(np.random.randn(1, 1, 150, 1)    * 0.01, dtype=tf.float32)
    w     = tf.Variable(np.random.randn(3, 3, 1, 10)    * 0.01, dtype=tf.float32)
    # feedback weights from v layer into q layer (~transition probabilities in Bellman equation)
    w_fb  = tf.Variable(np.random.randn(3, 3, 1, 10)    * 0.01, dtype=tf.float32)
    w_o   = tf.Variable(np.random.randn(10, 8)          * 0.01, dtype=tf.float32)

    # initial conv layer over image+reward prior
    h = conv2d_flipkernel(self.X, w0, name="h0") + bias

    r = conv2d_flipkernel(h, w1, name="r")
    q = conv2d_flipkernel(r, w, name="q")
    v = tf.reduce_max(q, axis=3, keep_dims=True, name="v")

    for i in range(0, k-1):
      rv = tf.concat([r, v], 3)
      wwfb = tf.concat([w, w_fb], 2)
      q = conv2d_flipkernel(rv, wwfb, name="q")
      v = tf.reduce_max(q, axis=3, keep_dims=True, name="v")

    # do one last convolution
    q = conv2d_flipkernel(tf.concat([r, v], 3),
                          tf.concat([w, w_fb], 2), name="q")

    # CHANGE TO THEANO ORDERING
    # Since we are selecting over channels, it becomes easier to work with
    # the tensor when it is in NCHW format vs NHWC
    q = tf.transpose(q, perm=[0, 3, 1, 2])

    # Select the conv-net channels at the state position (S1,S2).
    # This intuitively corresponds to each channel representing an action, and the convnet the Q function.
    # The tricky thing is we want to select the same (S1,S2) position *for each* channel and for each sample
    # TODO: performance can be improved here by substituting expensive
    #       transpose calls with better indexing for gather_nd
    bs = tf.shape(q)[0]
    rprn = tf.reshape(tf.tile(tf.reshape(tf.range(bs), [-1, 1]), [1, state_batch_size]), [-1])
    ins1 = tf.cast(tf.reshape(self.S1, [-1]), tf.int32)
    ins2 = tf.cast(tf.reshape(self.S2, [-1]), tf.int32)
    idx_in = tf.transpose(tf.stack([ins1, ins2, rprn]), [1, 0])
    q_out = tf.gather_nd(tf.transpose(q, [2, 3, 0, 1]), idx_in, name="q_out")

    # add logits
    self.logits = tf.matmul(q_out, w_o)
    # softmax output weights
    self.nn = tf.nn.softmax(self.logits, name="output")

  def build_loss(self):
    # Define loss and optimizer
    y_ = tf.cast(self.y, tf.int64)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=self.logits, labels=y_, name='cross_entropy')
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy_mean')
    tf.add_to_collection('losses', cross_entropy_mean)

    self.loss = tf.add_n(tf.get_collection('losses'), name='total_loss')
    cp = tf.cast(tf.argmax(self.nn, 1), tf.int32)
    self.err = tf.reduce_mean(tf.cast(tf.not_equal(cp, self.y), dtype=tf.float32))

  def build_optimizer(self):
    opt = tf.train.RMSPropOptimizer(learning_rate = FLAGS.LEARNING_RATE, epsilon = 1e-6, centered = True)
    self.optimizer = opt.minimize(self.loss)

  def build_summary(self):
    if True:
      for var in tf.trainable_variables():
        tf.summary.histogram(var.op.name, var)
      self.summary_op = tf.summary.merge_all()
      self.summary_writer = tf.summary.FileWriter(FLAGS.logdir)

  def train(self):
    print(fmt_row(10, ["Epoch", "Train Cost", "Train Err", "Epoch Time"]))
    for epoch in range(int(FLAGS.epochs)):
      tstart = time.time()
      avg_err, avg_cost = 0.0, 0.0
      num_batches = int(self.env.Xtrain.shape[0]/ FLAGS.batch_size)
      # Loop over all batches
      for i in range(0, self.env.Xtrain.shape[0], FLAGS.batch_size):
        j = i + FLAGS.batch_size
        if j <= self.env.Xtrain.shape[0]:
          # Run optimization op (backprop) and cost op (to get loss value)
          fd = {self.X: self.env.Xtrain[i:j], self.S1: self.env.S1train[i:j], self.S2: self.env.S2train[i:j],
              self.y: self.env.ytrain[i * FLAGS.statebatchsize:j * FLAGS.statebatchsize]}
          _, e_, c_ = self.sess.run([self.optimizer, self.err, self.loss], feed_dict=fd)
          avg_err += e_
          avg_cost += c_
      # Display logs per epoch step
      if epoch % 1 == 0:
        elapsed = time.time() - tstart
        print(fmt_row(10, [epoch, avg_cost/num_batches, avg_err/num_batches, elapsed]))
      if True:
        summary = tf.Summary()
        summary.ParseFromString(self.sess.run(self.summary_op))
        summary.value.add(tag='Average error', simple_value=float(avg_err/num_batches))
        summary.value.add(tag='Average cost', simple_value=float(avg_cost/num_batches))
        self.summary_writer.add_summary(summary, epoch)
    print("Finished training!")

  def test(self):
    correct_prediction = tf.cast(tf.argmax(self.nn, 1), tf.int32)
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(tf.not_equal(correct_prediction, self.y), dtype=tf.float32))
    acc = self.sess.run(accuracy, feed_dict = {self.X: self.env.Xtest, self.S1: self.env.S1test, self.S2: self.env.S2test, self.y: self.env.ytest})
    print("Accuracy: {}%".format(100 * (1 - acc)))
