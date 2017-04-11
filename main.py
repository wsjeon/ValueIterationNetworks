import time
import numpy as np
import tensorflow as tf
from data  import process_gridworld_data
from model import Agent
from utils import fmt_row

import os; os.environ["CUDA_VISIBLE_DEVICES"]="2"

# Data
tf.app.flags.DEFINE_string('input',           'data/gridworld_8.mat', 'Path to data')
tf.app.flags.DEFINE_integer('imsize',         8,                      'Size of input image')
# Parameters
tf.app.flags.DEFINE_float('lr',               0.001,                  'Learning rate for RMSProp')
tf.app.flags.DEFINE_integer('epochs',         30,                     'Maximum epochs to train for')
tf.app.flags.DEFINE_integer('k',              10,                     'Number of value iterations')
tf.app.flags.DEFINE_integer('ch_i',           2,                      'Channels in input layer')
tf.app.flags.DEFINE_integer('ch_h',           150,                    'Channels in initial hidden layer')
tf.app.flags.DEFINE_integer('ch_q',           10,                     'Channels in q layer (~actions)')
tf.app.flags.DEFINE_integer('batchsize',      12,                     'Batch size')
tf.app.flags.DEFINE_integer('statebatchsize', 10,                     'Number of state inputs for each sample (real number, technically is k+1)')
# Misc.
tf.app.flags.DEFINE_integer('seed',           0,                      'Random seed for numpy')
tf.app.flags.DEFINE_integer('display_step',   1,                      'Print summary output every n epochs')
tf.app.flags.DEFINE_boolean('log',            True,                  'Enable for tensorboard summary')
tf.app.flags.DEFINE_string('logdir',          './tmp/vintf/',          'Directory to store tensorboard summary')

agent = Agent()
