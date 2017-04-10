import time
import numpy as np
import tensorflow as tf
from data import process_gridworld_data
#from model import VI_Block
from model import Agent, Environment
from utils import fmt_row

import os; os.environ["CUDA_VISIBLE_DEVICES"]="0"

flags = tf.app.flags

flags.DEFINE_string('input',           'data/gridworld_8.mat', 'Path to data')
flags.DEFINE_integer('imsize',         8,                      'Size of input image')
flags.DEFINE_float('lr',               0.001,                  'Learning rate for RMSProp')
flags.DEFINE_integer('epochs',         30,                     'Maximum epochs to train for')
flags.DEFINE_integer('k',              10,                     'Number of value iterations')
flags.DEFINE_integer('ch_i',           2,                      'Channels in input layer')
flags.DEFINE_integer('ch_h',           150,                    'Channels in initial hidden layer')
flags.DEFINE_integer('ch_q',           10,                     'Channels in q layer (~actions)')
flags.DEFINE_integer('batch_size',      12,                     'Batch size')
flags.DEFINE_integer('statebatchsize', 10,                     'Number of state inputs for each sample (real number, technically is k+1)')
flags.DEFINE_integer('seed',           0,                      'Random seed for numpy')
flags.DEFINE_integer('display_step',   1,                      'Print summary output every n epochs')
flags.DEFINE_boolean('log',            False,                  'Enable for tensorboard summary')
flags.DEFINE_string('logdir',          './tmp',          'Directory to store tensorboard summary')

FLAGS = tf.app.flags.FLAGS

np.random.seed(FLAGS.seed)

## symbolic input image tensor where typically first channel is image, second is the reward prior
#X  = tf.placeholder(tf.float32, name="X",  shape=[None, FLAGS.imsize, FLAGS.imsize, FLAGS.ch_i])
## symbolic input batches of vertical positions
#S1 = tf.placeholder(tf.int32,   name="S1", shape=[None, FLAGS.statebatchsize])
## symbolic input batches of horizontal positions
#S2 = tf.placeholder(tf.int32,   name="S2", shape=[None, FLAGS.statebatchsize])
#y  = tf.placeholder(tf.int32,   name="y",  shape=[None])
#

# Load data
#Xtrain, S1train, S2train, ytrain, Xtest, S1test, S2test, ytest = process_gridworld_data(input=FLAGS.input, imsize=FLAGS.imsize)
env = Environment()

# Construct model (Value Iteration Network)
agent = Agent(env)
#logits, nn = agent.VI_Block(X, S1, S2)

## Define loss and optimizer
#y_ = tf.cast(y, tf.int64)
#cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
#    logits=logits, labels=y_, name='cross_entropy')
#cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy_mean')
#tf.add_to_collection('losses', cross_entropy_mean)
#
#cost = tf.add_n(tf.get_collection('losses'), name='total_loss')
#optimizer = tf.train.RMSPropOptimizer(learning_rate=FLAGS.lr, epsilon=1e-6, centered=True).minimize(cost)
#

## Test model & calculate accuracy
#cp = tf.cast(tf.argmax(agent.nn, 1), tf.int32)
#err = tf.reduce_mean(tf.cast(tf.not_equal(cp, agent.y), dtype=tf.float32))

## Launch the graph
#if FLAGS.log:
#  for var in tf.trainable_variables():
#    tf.summary.histogram(var.op.name, var)
#  summary_op = tf.summary.merge_all()
#  summary_writer = tf.summary.FileWriter(FLAGS.logdir, sess.graph)
agent.train()
#print(fmt_row(10, ["Epoch", "Train Cost", "Train Err", "Epoch Time"]))
#for epoch in range(int(FLAGS.epochs)):
#  tstart = time.time()
#  avg_err, avg_cost = 0.0, 0.0
#  num_batches = int(Xtrain.shape[0]/FLAGS.batch_size)
#  # Loop over all batches
#  for i in range(0, Xtrain.shape[0], FLAGS.batch_size):
#    j = i + FLAGS.batch_size
#    if j <= Xtrain.shape[0]:
#      # Run optimization op (backprop) and cost op (to get loss value)
#      fd = {agent.X: Xtrain[i:j], agent.S1: S1train[i:j], agent.S2: S2train[i:j],
#          agent.y: ytrain[i * FLAGS.statebatchsize:j * FLAGS.statebatchsize]}
#      _, e_, c_ = sess.run([agent.optimizer, err, agent.loss], feed_dict=fd)
#      avg_err += e_
#      avg_cost += c_
#  # Display logs per epoch step
#  if epoch % FLAGS.display_step == 0:
#    elapsed = time.time() - tstart
#    print(fmt_row(10, [epoch, avg_cost/num_batches, avg_err/num_batches, elapsed]))
#  if FLAGS.log:
#    summary = tf.Summary()
#    summary.ParseFromString(sess.run(summary_op))
#    summary.value.add(tag='Average error', simple_value=float(avg_err/num_batches))
#    summary.value.add(tag='Average cost', simple_value=float(avg_cost/num_batches))
#    summary_writer.add_summary(summary, epoch)
#print("Finished training!")

## Test model
#correct_prediction = tf.cast(tf.argmax(agent.nn, 1), tf.int32)
## Calculate accuracy
#accuracy = tf.reduce_mean(tf.cast(tf.not_equal(correct_prediction, agent.y), dtype=tf.float32))
#acc = accuracy.eval({agent.X: Xtest, agent.S1: S1test, agent.S2: S2test, agent.y: ytest})
#print("Accuracy: {}%".format(100 * (1 - acc)))
agent.test()
agent.sess.close()
