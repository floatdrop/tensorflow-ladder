import tensorflow as tf
from batch_normalization import batch_norm

class _Record:
  pass

def _build_data_placeholders(input_unit_count, class_count):
  placeholders = _Record()
  placeholders.inputs = tf.placeholder(tf.float32, [None, input_unit_count], name = 'inputs')
  placeholders.labels = tf.placeholder(tf.float32, [None, class_count], name = 'labels')
  placeholders.training_phase = tf.placeholder(tf.bool, name = 'training_phase')
  return placeholders

def _weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev = 0.1)
  return tf.Variable(initial)

def _bias_variable(shape):
  initial = tf.constant(0.1, shape = shape)
  return tf.Variable(initial)

def _fully_connected_relu_layer(inputs, unit_count, training_phase):
  input_unit_count = inputs.get_shape()[1].value
  weights = _weight_variable([input_unit_count, unit_count])
  linear = batch_norm(tf.matmul(inputs, weights), training_phase = training_phase)
  return tf.nn.relu(linear)

def _softmax_layer(inputs, unit_count, training_phase):
  input_unit_count = inputs.get_shape()[1].value
  weights = _weight_variable([input_unit_count, unit_count])
  linear = batch_norm(tf.matmul(inputs, weights), training_phase = training_phase)
  return tf.nn.softmax(linear)

def _build_encoder_layers(placeholders, layer_sizes):
  outputs = []
  previous_output = placeholders.inputs
  for layer_size in layer_sizes:
    previous_output = _fully_connected_relu_layer(previous_output, layer_size, placeholders.training_phase)
    outputs.append(previous_output)
  return outputs

def _build_forward_pass(placeholders):
  class_count = placeholders.labels.get_shape()[1].value

  encoder_outputs = _build_encoder_layers(placeholders, [100])
  softmax_output = _softmax_layer(encoder_outputs[-1], class_count, placeholders.training_phase)

  output = _Record()
  output.label_probabilities = softmax_output
  return output

def _build_train_step(placeholders, output, learning_rate):
  cross_entropy = -tf.reduce_sum(placeholders.labels * tf.log(output.label_probabilities))
  return tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)

def _build_accuracy_measure(placeholders, output):
  correct_prediction = tf.equal(tf.argmax(output.label_probabilities, 1), tf.argmax(placeholders.labels, 1))
  return tf.reduce_mean(tf.cast(correct_prediction, "float"))

class Model:
  def __init__(self, input_unit_count, class_count):
    learning_rate = 0.001

    self.placeholders = _build_data_placeholders(input_unit_count, class_count)
    self.output = _build_forward_pass(self.placeholders)
    self.train_step = _build_train_step(self.placeholders, self.output, learning_rate)
    self.accuracy_measure = _build_accuracy_measure(self.placeholders, self.output)

  def fill_placeholders(self, inputs = None, labels = None, training_phase = True):
    replacements = {}
    if inputs is not None:
      replacements[self.placeholders.inputs] = inputs
    if labels is not None:
      replacements[self.placeholders.labels] = labels
    replacements[self.placeholders.training_phase] = training_phase
    return replacements

class Session:
  def __init__(self, model):
    self.session = tf.Session()
    self.model = model

  def __enter__(self):
    self.session.run(tf.initialize_all_variables())
    return self

  def __exit__(self, type, value, traceback):
    self.session.close()

  def train_batch(self, inputs, labels):
    return self.session.run(self.model.train_step, self.model.fill_placeholders(inputs, labels, training_phase = True))

  def test(self, inputs, labels):
    return self.session.run(self.model.accuracy_measure, self.model.fill_placeholders(inputs, labels, training_phase = False))
