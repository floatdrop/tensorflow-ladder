import tensorflow as tf
import numpy
from tensorflow.python import control_flow_ops
from batch_normalization import batch_norm
from time import strftime

class _Record:
  pass

def _build_data_placeholders(input_layer_size, class_count):
  placeholders = _Record()
  placeholders.inputs = tf.placeholder(tf.float32, [None, input_layer_size], name = 'inputs')
  placeholders.labels = tf.placeholder(tf.float32, [None, class_count], name = 'labels')
  placeholders.is_training_phase = tf.placeholder(tf.bool, name = 'is_training_phase')
  return placeholders

def _weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev = 0.1)
  return tf.Variable(initial)

def _bias_variable(shape):
  initial = tf.constant(0.1, shape = shape)
  return tf.Variable(initial)

def _layer_size(layer_output):
  return layer_output.get_shape()[1].value

def _fully_connected_layer(inputs, output_size, non_linearity, is_training_phase):
  weights = _weight_variable([_layer_size(inputs), output_size])
  linear = batch_norm(tf.matmul(inputs, weights), is_training_phase = is_training_phase)
  return non_linearity(linear)

def _build_encoder_layers(input_layer, other_layer_definitions, is_training_phase):
  layer_outputs = [input_layer]
  for (layer_size, non_linearity) in other_layer_definitions:
    layer_output = _fully_connected_layer(
      inputs = layer_outputs[-1],
      output_size = layer_size,
      non_linearity = non_linearity,
      is_training_phase = is_training_phase)
    layer_outputs.append(layer_output)
  return layer_outputs

def _build_decoder_layers(encoder_layers, is_training_phase):
  layer_outputs = [encoder_layers[-1]]
  for encoder_layer in reversed(encoder_layers[:-1]):
    layer_output = _fully_connected_layer(
      inputs = layer_outputs[-1],
      output_size = _layer_size(encoder_layer),
      non_linearity = tf.nn.relu,
      is_training_phase = is_training_phase)
    layer_outputs.append(layer_output)
  return layer_outputs

def _build_forward_pass(placeholders):
  encoder_layer_definitions = [
    (100, tf.nn.relu),
    (50, tf.nn.relu),
    (_layer_size(placeholders.labels), tf.nn.softmax)
  ]
  encoder_outputs = _build_encoder_layers(
    input_layer = placeholders.inputs,
    other_layer_definitions = encoder_layer_definitions,
    is_training_phase = placeholders.is_training_phase)

  decoder_outputs = _build_decoder_layers(
    encoder_layers = encoder_outputs,
    is_training_phase = placeholders.is_training_phase)

  output = _Record()
  output.label_probabilities = encoder_outputs[-1]
  output.autoencoded_inputs = decoder_outputs[-1]
  return output

def _autoencoder_cost(placeholders, output, summary_tag):
  cost = tf.reduce_mean(tf.pow(placeholders.inputs - output.autoencoded_inputs, 2))
  tf.scalar_summary("autoencoder cost (%s)" % summary_tag, cost)
  return cost

def _cost_entropy(placeholders, output):
  cross_entropy = -tf.reduce_mean(
    placeholders.labels * tf.log(output.label_probabilities))
  tf.scalar_summary("cross entropy", cross_entropy)
  return cross_entropy

def _build_supervised_train_step(placeholders, output, learning_rate, cross_entropy_training_weight):
  cross_entropy = _cost_entropy(placeholders, output)
  autoencoder_cost = _autoencoder_cost(placeholders, output, "supervised")
  total_cost = cross_entropy_training_weight * cross_entropy + autoencoder_cost
  return tf.train.GradientDescentOptimizer(learning_rate).minimize(total_cost)

def _build_unsupervised_train_step(placeholders, output, learning_rate):
  autoencoder_cost = _autoencoder_cost(placeholders, output, "unsupervised")
  return tf.train.GradientDescentOptimizer(learning_rate).minimize(autoencoder_cost)

def _build_accuracy_measure(placeholders, output):
  correct_prediction = tf.equal(tf.argmax(output.label_probabilities, 1), tf.argmax(placeholders.labels, 1))
  return tf.reduce_mean(tf.cast(correct_prediction, "float"))

class Model:
  def __init__(self, input_layer_size, class_count):
    learning_rate = 0.01
    cross_entropy_training_weight = 3

    self.placeholders = _build_data_placeholders(input_layer_size, class_count)
    self.output = _build_forward_pass(self.placeholders)
    self.supervised_train_step = _build_supervised_train_step(
      self.placeholders, self.output, learning_rate, cross_entropy_training_weight)
    self.unsupervised_train_step = _build_unsupervised_train_step(self.placeholders, self.output, learning_rate)
    self.accuracy_measure = _build_accuracy_measure(self.placeholders, self.output)

  def fill_placeholders(self, inputs, labels = None, is_training_phase = True):
    if labels is None:
      labels = numpy.zeros([inputs.shape[0], _layer_size(self.placeholders.labels)])
    return {
      self.placeholders.inputs: inputs,
      self.placeholders.labels: labels,
      self.placeholders.is_training_phase: is_training_phase
    }

class Session:
  def __init__(self, model):
    self.session = tf.Session()
    self.model = model
    self.summaries = tf.merge_all_summaries()
    self.writer = tf.train.SummaryWriter(strftime("logs/%Y-%m-%d_%H:%M:%S"))

  def __enter__(self):
    self.session.run(tf.initialize_all_variables())
    return self

  def __exit__(self, type, value, traceback):
    self.session.close()

  def train_supervised_batch(self, inputs, labels, step_number):
    train_result, summary = self.session.run(
      [self.model.supervised_train_step, self.summaries],
      self.model.fill_placeholders(
        inputs, labels, is_training_phase = True))

    self.writer.add_summary(summary, step_number)
    return train_result

  def train_unsupervised_batch(self, inputs, step_number):
    train_result, summary = self.session.run(
      [self.model.unsupervised_train_step, self.summaries],
      self.model.fill_placeholders(inputs, is_training_phase = True))

    self.writer.add_summary(summary, step_number)
    return train_result

  def test(self, inputs, labels):
    return self.session.run(self.model.accuracy_measure,
      self.model.fill_placeholders(inputs, labels, is_training_phase = False))
