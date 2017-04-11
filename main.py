import tensorflow as tf
import numpy as np
from model import Agent, Environment

import os; os.environ["CUDA_VISIBLE_DEVICES"]="0"

flags = tf.app.flags

flags.DEFINE_string('input',           'data/gridworld_8.mat', 'Path to data')
flags.DEFINE_integer('MAZE_SIZE',         8,                      'Size of input image')
flags.DEFINE_float('LEARNING_RATE',               0.001,                  'Learning rate for RMSProp')
flags.DEFINE_integer('epochs',         30,                     'Maximum epochs to train for')
flags.DEFINE_integer('k',              10,                     'Number of value iterations')
flags.DEFINE_integer('batch_size',      12,                     'Batch size')
flags.DEFINE_integer('statebatchsize', 10,                     'Number of state inputs for each sample (real number, technically is k+1)')
flags.DEFINE_string('logdir',          './tmp',          'Directory to store tensorboard summary')

FLAGS = tf.app.flags.FLAGS

np.random.seed(0)

# Load data
env = Environment()

# Construct model (Value Iteration Network)
agent = Agent(env)
agent.train()
agent.test()
agent.sess.close()
