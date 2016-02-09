## Adapted from http://stackoverflow.com/a/34634291/64979

import tensorflow as tf
from tensorflow.python import control_flow_ops

def batch_norm(inputs, is_training_phase):
  """
  Batch normalization for fully connected layers.
  Args:
    inputs:            2D Tensor, batch size * layer width
    is_training_phase: boolean tf.Variable, true indicates training phase
  Return:
    normed:            batch-normalized Tensor
  """
  with tf.name_scope('batch_norm') as scope:
    depth = inputs.get_shape()[-1].value

    batch_mean, batch_var = tf.nn.moments(inputs, [0], name = 'moments')
    batch_std = tf.sqrt(batch_var)
    ema = tf.train.ExponentialMovingAverage(decay = 0.9)
    ema_apply_op = ema.apply([batch_mean, batch_var])
    ema_mean, ema_var = ema.average(batch_mean), ema.average(batch_var)

    def mean_var_with_update():
      with tf.control_dependencies([ema_apply_op]):
        return tf.identity(batch_mean), tf.identity(batch_var)

    mean, var = control_flow_ops.cond(is_training_phase,
      mean_var_with_update,
      lambda: (ema_mean, ema_var))

    normed = (inputs - batch_mean) / batch_std

    return normed, batch_mean, batch_std
