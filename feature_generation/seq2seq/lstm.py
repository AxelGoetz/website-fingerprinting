"""
TODO: Finish
"""

import math
import numpy as np
import tensorflow as tf
from tensorflow.contrib.rnn import LSTMCell, GRUCell

class BNLSTMCell(LSTMCell):
    '''Batch normalized LSTM as described in arxiv.org/abs/1603.09025'''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, inputs, state, scope=None, training=True):
        with tf.variable_scope('BatchNorm') as scope:
            bn = tf.contrib.layers.batch_norm(inputs, center=True, scale=True, is_training=True, scope=scope)
        import pdb; pdb.set_trace()

        return super().__call__(bn, state, scope)

class BNGRUCell(GRUCell):
    '''Batch normalized GRU as described in arxiv.org/abs/1603.09025'''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, inputs, state, scope=None, training=True):
        bn = tf.contrib.layers.batch_norm(inputs, center=True, scale=True, is_training=training, scope='bn')

        return super().__call__(bn, state, scope)
