import time
import numpy as np
import tensorflow as tf
from data  import process_gridworld_data
from model import Agent
from utils import fmt_row

import os; os.environ['CUDA_VISIBLE_DEVICES']='0'

flags = tf.app.flags

flags.DEFINE_string('DATA_PATH', './data/gridworld_8.mat', 'training data path')
flags.DEFINE_integer('IM_SIZE', 8, 'size of maze image')
flags.DEFINE_float('LR', 0.001, 'learning rate')
flags.DEFINE_integer('MAX_EPOCHS', 30, 'maximum epoch for training network')
flags.DEFINE_integer('NUM_VI', 10, 'Number of value iterations')
flags.DEFINE_integer('batchsize',      12,                     'Batch size')
flags.DEFINE_integer('NUM_STATES', 10,
    'number of states for each sampled maze (technically, NUM_VI + 1)')
flags.DEFINE_integer('SEED', 0, 'random seed for reproducibility')
flags.DEFINE_string('LOGDIR', './tmp', 'path to save event files')

def main():
  """ Agent setting """
  agent = Agent()

  """ Learning """
  agent.learn()

  """ Test """
  agent.test()
  agent.sess.close()

if __name__ == "__main__":
  main()
