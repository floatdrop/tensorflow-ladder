import tensorflow as tf

class _Record:
  pass

def _build_data_placeholders(input_dimension_count, label_count):
  placeholders = _Record()
  placeholders.inputs = tf.placeholder(tf.float32, [None, input_dimension_count])
  placeholders.labels = tf.placeholder(tf.float32, [None, label_count])
  return placeholders

def _build_forward_pass(placeholders):
  input_dimension_count = placeholders.inputs.get_shape()[1].value
  label_count = placeholders.labels.get_shape()[1].value

  outputs = _Record()
  weights = tf.Variable(tf.zeros([input_dimension_count, label_count]))
  biases = tf.Variable(tf.zeros([label_count]))
  hidden_activations = tf.matmul(placeholders.inputs, weights) + biases
  outputs.label_probabilities = tf.nn.softmax(hidden_activations)
  return outputs

def _build_train_step(placeholders, outputs, learning_rate):
  cross_entropy = -tf.reduce_sum(placeholders.labels * tf.log(outputs.label_probabilities))
  return tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)

def _build_accuracy_measure(placeholders, outputs):
  correct_prediction = tf.equal(tf.argmax(outputs.label_probabilities, 1), tf.argmax(placeholders.labels, 1))
  return tf.reduce_mean(tf.cast(correct_prediction, "float"))

class Model:
  def __init__(self, input_dimension_count, label_count):
    learning_rate = 0.01

    self.placeholders = _build_data_placeholders(input_dimension_count, label_count)
    self.outputs = _build_forward_pass(self.placeholders)
    self.train_step = _build_train_step(self.placeholders, self.outputs, learning_rate)
    self.accuracy_measure = _build_accuracy_measure(self.placeholders, self.outputs)
    self.sample = tf.argmax(self.outputs.label_probabilities, 1)

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

  def sample(self, inputs):
    return self.session.run(self.model.sample, self.model.fill_placeholders(inputs))
