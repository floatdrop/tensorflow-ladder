## Adapted from http://stackoverflow.com/a/34634291/64979

import tensorflow as tf
from tensorflow.python import control_flow_ops

def batch_norm(inputs, training_phase, scope = 'bn'):
  """
  Batch normalization for fully connected layers.
  Args:
    inputs:         2D Tensor, batch size * layer width
    training_phase: boolean tf.Variable, true indicates training phase
    scope:          string, variable scope
  Return:
    normed:         batch-normalized map
  """
  depth = inputs.get_shape()[-1].value
  inputs_4d = tf.reshape(inputs, [-1, 1, 1, depth])

  with tf.variable_scope(scope):
    beta = tf.Variable(tf.constant(0.0, shape = [depth]),
      name = 'beta', trainable = True)
    gamma = tf.Variable(tf.constant(1.0, shape = [depth]),
      name = 'gamma', trainable = True)

    batch_mean, batch_var = tf.nn.moments(inputs_4d, [0, 1, 2], name = 'moments')
    ema = tf.train.ExponentialMovingAverage(decay = 0.9)
    ema_apply_op = ema.apply([batch_mean, batch_var])
    ema_mean, ema_var = ema.average(batch_mean), ema.average(batch_var)

    def mean_var_with_update():
      with tf.control_dependencies([ema_apply_op]):
        return tf.identity(batch_mean), tf.identity(batch_var)

    mean, var = control_flow_ops.cond(training_phase,
      mean_var_with_update,
      lambda: (ema_mean, ema_var))

    normed_4d = tf.nn.batch_norm_with_global_normalization(inputs_4d, mean, var,
      beta, gamma, 1e-3, scale_after_normalization = True)
    normed = tf.reshape(normed_4d, [-1, depth])

  return normed
