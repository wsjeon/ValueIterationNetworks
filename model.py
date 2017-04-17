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
  def __init__(self):
    self.FLAGS = tf.app.flags.FLAGS
    """ Fix random seed for reproducibility """
#    np.random.seed(self.FLAGS.SEED)
#    tf.set_random_seed(self.FLAGS.SEED)

    """ TensorFlow graph construction """
    self.build_model()
    self.build_loss()
    self.build_optimizer()
    self.build_summary()

    """ Open TensorFlow session and initialize variables. """
    self.sess = tf.Session()
    self.sess.run(tf.global_variables_initializer())

  def build_model(self):
    FLAGS = self.FLAGS
    self.X = tf.placeholder(tf.float32, [None, FLAGS.IM_SIZE, FLAGS.IM_SIZE, 2])
    self.S1 = tf.placeholder(tf.int32, [None, FLAGS.NUM_STATES])
    self.S2 = tf.placeholder(tf.int32, [None, FLAGS.NUM_STATES])

    init = tf.random_normal_initializer(stddev = 0.01)

    with slim.arg_scope([conv2d, fc], activation_fn = None, weights_initializer = init,
        biases_initializer = None):
      h = conv2d(self.X, 150, [3, 3], biases_initializer = init, scope = "conv1")
      r = conv2d(h, 1, [3, 3], scope = "conv2")
      v = tf.zeros(tf.shape(r))
      for i in range(FLAGS.NUM_VI + 1): # totally, NUM_VI value iterations and get q.
        q = conv2d(tf.concat([r, v], axis = 3), 10, [3, 3], reuse = True, scope = "conv3")
        v = tf.reduce_max(q, axis = 3, keep_dims = True)

      X_idx = tf.reshape(tf.range(tf.shape(q)[0]), [-1, 1])
      q_idx = tf.stack([
        tf.reshape(tf.tile(X_idx, [1, FLAGS.NUM_STATES]), [-1]),
        tf.reshape(self.S1, [-1]),
        tf.reshape(self.S2, [-1])],
        axis = 1)
      q_out = tf.gather_nd(q, q_idx)

      # add logits
      self.logits = fc(q_out, 8, scope = "fc0")
      # softmax output weights
      self.net = tf.nn.softmax(self.logits)

  def build_loss(self):
    self.y = tf.placeholder(tf.int32, [None])
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits = self.logits, labels = self.y)
    self.loss = tf.reduce_mean(cross_entropy)

  def build_optimizer(self):
    optimizer = tf.train.RMSPropOptimizer(learning_rate = self.FLAGS.LR,
        epsilon = 1e-6, centered = True)
    self.optimizer = optimizer.minimize(self.loss)

  def build_summary(self):
    self.cp = tf.cast(tf.argmax(self.net, axis = 1), tf.int32)
    self.err = tf.reduce_mean(tf.cast(tf.not_equal(self.cp, self.y), dtype=tf.float32))
    for var in tf.trainable_variables():
      tf.summary.histogram(var.op.name, var)
    self.summary_op = tf.summary.merge_all()
    self.summary_writer = tf.summary.FileWriter(self.FLAGS.LOGDIR)

  def learn(self):
    FLAGS = self.FLAGS
    Xtrain, S1train, S2train, ytrain, _, _, _, _ =\
        process_gridworld_data(input = FLAGS.DATA_PATH, imsize=FLAGS.IM_SIZE)

    batch_size = FLAGS.batchsize
    print(fmt_row(10, ["Epoch", "Train Cost", "Train Err", "Epoch Time"]))
    for epoch in range(int(FLAGS.MAX_EPOCHS)):
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
                      self.y: ytrain[i * FLAGS.NUM_STATES:j * FLAGS.NUM_STATES]}
                _, e_, c_ = self.sess.run([self.optimizer, self.err, self.loss], feed_dict=fd)
                avg_err += e_
                avg_cost += c_
        # Display logs per epoch step
        elapsed = time.time() - tstart
        print(fmt_row(10, [epoch, avg_cost/num_batches, avg_err/num_batches, elapsed]))
        summary = tf.Summary()
        summary.ParseFromString(self.sess.run(self.summary_op))
        summary.value.add(tag='Average error', simple_value=float(avg_err/num_batches))
        summary.value.add(tag='Average cost', simple_value=float(avg_cost/num_batches))
        self.summary_writer.add_summary(summary, epoch)
    print("Finished training!")

  def test(self):
    FLAGS = self.FLAGS
    _, _, _, _, Xtest, S1test, S2test, ytest =\
        process_gridworld_data(input = FLAGS.DATA_PATH, imsize=FLAGS.IM_SIZE)
    # Test model
    correct_prediction = tf.cast(tf.argmax(self.net, 1), tf.int32)
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(tf.not_equal(correct_prediction, self.y), dtype=tf.float32))
    acc = self.sess.run(accuracy,
        feed_dict = {self.X: Xtest, self.S1: S1test, self.S2: S2test, self.y: ytest})
    print "Accuracy: {}%".format(100 * (1 - acc))
