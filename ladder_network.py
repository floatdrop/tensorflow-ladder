import tensorflow as tf

class _Record:
  pass

def _build_data_placeholders(input_unit_count, class_count):
  placeholders = _Record()
  placeholders.inputs = tf.placeholder(tf.float32, [None, input_unit_count])
  placeholders.labels = tf.placeholder(tf.float32, [None, class_count])
  return placeholders

def _weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev = 0.1)
  return tf.Variable(initial)

def _bias_variable(shape):
  initial = tf.constant(0.1, shape = shape)
  return tf.Variable(initial)

def _fully_connected_relu_layer(inputs, unit_count):
  input_unit_count = inputs.get_shape()[1].value

  weights = _weight_variable([input_unit_count, unit_count])
  biases = _bias_variable([unit_count])
  return tf.nn.relu(tf.matmul(inputs, weights) + biases)

def _softmax_layer(inputs, unit_count):
  input_unit_count = inputs.get_shape()[1].value
  
  weights = _weight_variable([input_unit_count, unit_count])
  biases = _bias_variable([unit_count])
  return tf.nn.softmax(tf.matmul(inputs, weights) + biases)

def _build_forward_pass(placeholders):
  input_unit_count = placeholders.inputs.get_shape()[1].value
  class_count = placeholders.labels.get_shape()[1].value

  activations1 = _fully_connected_relu_layer(placeholders.inputs, 100)
  activations2 = _softmax_layer(activations1, 10)

  outputs = _Record()
  outputs.label_probabilities = activations2
  return outputs

def _build_train_step(placeholders, outputs, learning_rate):
  cross_entropy = -tf.reduce_sum(placeholders.labels * tf.log(outputs.label_probabilities))
  return tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)

def _build_accuracy_measure(placeholders, outputs):
  correct_prediction = tf.equal(tf.argmax(outputs.label_probabilities, 1), tf.argmax(placeholders.labels, 1))
  return tf.reduce_mean(tf.cast(correct_prediction, "float"))

class Model:
  def __init__(self, input_unit_count, class_count):
    learning_rate = 0.001

    self.placeholders = _build_data_placeholders(input_unit_count, class_count)
    self.outputs = _build_forward_pass(self.placeholders)
    self.train_step = _build_train_step(self.placeholders, self.outputs, learning_rate)
    self.accuracy_measure = _build_accuracy_measure(self.placeholders, self.outputs)

  def fill_placeholders(self, inputs = None, labels = None):
    replacements = {}
    if inputs is not None:
      replacements[self.placeholders.inputs] = inputs
    if labels is not None:
      replacements[self.placeholders.labels] = labels
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
    return self.session.run(self.model.train_step, self.model.fill_placeholders(inputs, labels))

  def test(self, inputs, labels):
    return self.session.run(self.model.accuracy_measure, self.model.fill_placeholders(inputs, labels))
