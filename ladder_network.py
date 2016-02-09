import tensorflow as tf
import numpy
from tensorflow.python import control_flow_ops
from batch_normalization import batch_norm
from time import strftime

class Model:
  def __init__(self, input_layer_size, class_count):
    self.hyperparameters = {
      "learning_rate": 0.01,
      "cross_entropy_training_weight": 3,
      "noise_level": 0.2
    }

    self.placeholders = self._data_placeholders(
        input_layer_size, class_count)
    self.output = self._forward_pass(self.placeholders)
    self.supervised_train_step = self._supervised_train_step(
        self.placeholders, self.output)
    self.unsupervised_train_step = self._unsupervised_train_step(
        self.placeholders, self.output)
    self.accuracy_measure = self._accuracy_measure(
        self.placeholders, self.output)

    self.unsupervised_summaries = tf.merge_all_summaries("unsupervised")
    self.supervised_summaries = tf.merge_all_summaries("supervised")
    self.test_summaries = tf.merge_all_summaries("test")

  def fill_placeholders(self, inputs, labels = None, is_training_phase = True):
    if labels is None:
      labels = numpy.zeros([inputs.shape[0], self._layer_size(self.placeholders.labels)])
    return {
      self.placeholders.inputs: inputs,
      self.placeholders.labels: labels,
      self.placeholders.is_training_phase: is_training_phase
    }

  class _Record:
    pass

  def _data_placeholders(self, input_layer_size, class_count):
    with tf.name_scope("placeholders") as scope:
      placeholders = self._Record()
      placeholders.inputs = tf.placeholder(tf.float32, [None, input_layer_size], name = 'inputs')
      placeholders.labels = tf.placeholder(tf.float32, [None, class_count], name = 'labels')
      placeholders.is_training_phase = tf.placeholder(tf.bool, name = 'is_training_phase')
      return placeholders

  def _forward_pass(self, placeholders):
    encoder_layer_definitions = [
      (100, tf.nn.relu),
      (50, tf.nn.relu),
      (self._layer_size(placeholders.labels), tf.nn.softmax)
    ]
    encoder_outputs = self._encoder_layers(
        input_layer = placeholders.inputs,
        other_layer_definitions = encoder_layer_definitions,
        noise_level = self.hyperparameters["noise_level"],
        is_training_phase = placeholders.is_training_phase)

    decoder_outputs = self._decoder_layers(
        encoder_layers = encoder_outputs,
        is_training_phase = placeholders.is_training_phase)

    output = self._Record()
    output.label_probabilities = encoder_outputs[-1]
    output.autoencoded_inputs = decoder_outputs[-1]
    output.encoder_outputs = encoder_outputs
    output.decoder_outputs = decoder_outputs
    return output

  def _supervised_train_step(self, placeholders, output):
    with tf.name_scope("supervised_training") as scope:
      total_cost = self._total_cost(placeholders, output,
          self.hyperparameters["cross_entropy_training_weight"])
      return self._optimizer(self.hyperparameters["learning_rate"], total_cost)

  def _unsupervised_train_step(self, placeholders, output):
    with tf.name_scope("unsupervised_training") as scope:
      autoencoder_cost = self._autoencoder_cost(placeholders, output, "unsupervised")
      return self._optimizer(self.hyperparameters["learning_rate"], autoencoder_cost)

  def _accuracy_measure(self, placeholders, output):
    with tf.name_scope("accuracy_measure") as scope:
      correct_prediction = tf.equal(tf.argmax(output.label_probabilities, 1), tf.argmax(placeholders.labels, 1))
      accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
      tf.scalar_summary("test accuracy", accuracy, ["test"])
      return accuracy

  def _encoder_layers(self,
        input_layer, other_layer_definitions,
      noise_level, is_training_phase):
    with tf.name_scope("encoder") as scope:
      layer_outputs = [input_layer]
      for (layer_size, non_linearity) in other_layer_definitions:
        layer_output = self._fully_connected_layer(
            inputs = layer_outputs[-1],
            output_size = layer_size,
            non_linearity = non_linearity,
            noise_level = noise_level,
            is_training_phase = is_training_phase)
        layer_outputs.append(layer_output)
      return layer_outputs

  def _decoder_layers(self, encoder_layers, is_training_phase):
    with tf.name_scope("decoder") as scope:
      layer_outputs = [encoder_layers[-1]]
      for encoder_layer in reversed(encoder_layers[:-1]):
        layer_output = self._fully_connected_layer(
            inputs = layer_outputs[-1],
            output_size = self._layer_size(encoder_layer),
            non_linearity = tf.nn.relu,
            noise_level = 0.0,
            is_training_phase = is_training_phase)
        layer_outputs.append(layer_output)
      return layer_outputs

  def _fully_connected_layer(self,
      inputs, output_size, non_linearity,
      noise_level, is_training_phase):
    with tf.name_scope("layer") as scope:
      weights = self._weight_variable([self._layer_size(inputs), output_size])
      linear = batch_norm(tf.matmul(inputs, weights),
          is_training_phase = is_training_phase)
      corrupted = linear + tf.random_normal([output_size], mean = 0.0, stddev = noise_level)
      return non_linearity(linear)

  def _optimizer(self, learning_rate, cost_function):
    with tf.name_scope("optimizer") as scope:
      optimizer = tf.train.GradientDescentOptimizer(learning_rate)
      return optimizer.minimize(cost_function)

  def _total_cost(self, placeholders, output, cross_entropy_training_weight):
    with tf.name_scope("total_cost") as scope:
      cross_entropy = self._cost_entropy(placeholders, output)
      autoencoder_cost = self._autoencoder_cost(placeholders, output, "supervised")
      return cross_entropy_training_weight * cross_entropy + autoencoder_cost

  def _autoencoder_cost(self, placeholders, output, summary_tag):
    with tf.name_scope("autoencoder_cost") as scope:
      encoder_outputs = output.encoder_outputs
      decoder_outputs = list(reversed(output.decoder_outputs))

      assert all(encoder.get_shape().is_compatible_with(decoder.get_shape())
        for (encoder, decoder) in zip(encoder_outputs, decoder_outputs))

      layer_costs = [tf.reduce_mean(tf.pow(encoder - decoder, 2))
        for (encoder, decoder) in zip(encoder_outputs, decoder_outputs)]

      for index, layer_cost in enumerate(layer_costs):
        tf.scalar_summary("layer %i autoencoder cost" % index, layer_cost, [summary_tag])

      autoencoder_cost = sum(layer_costs)
      tf.scalar_summary("autoencoder cost", autoencoder_cost, [summary_tag])
      return autoencoder_cost

  def _cost_entropy(self, placeholders, output):
    with tf.name_scope("cross_entropy_cost") as scope:
      cross_entropy = -tf.reduce_mean(
          placeholders.labels * tf.log(output.label_probabilities))
      tf.scalar_summary("cross entropy", cross_entropy, ["supervised"])
      return cross_entropy

  def _weight_variable(self, shape):
    with tf.name_scope("weight") as scope:
      initial = tf.truncated_normal(shape, stddev = 0.1)
      return tf.Variable(initial)

  def _bias_variable(self, shape):
    with tf.name_scope("bias") as scope:
      initial = tf.constant(0.1, shape = shape)
      return tf.Variable(initial)

  def _layer_size(self, layer_output):
    return layer_output.get_shape()[1].value


class Session:
  def __init__(self, model):
    self.session = tf.Session()
    self.model = model
    self.writer = tf.train.SummaryWriter(
        logdir = strftime("logs/%Y-%m-%d_%H:%M:%S"),
        graph_def = self.session.graph_def)

  def __enter__(self):
    self.session.run(tf.initialize_all_variables())
    return self

  def __exit__(self, type, value, traceback):
    self.session.close()

  def train_supervised_batch(self, inputs, labels, step_number):
    train_result, summary = self.session.run(
      [self.model.supervised_train_step, self.model.supervised_summaries],
      self.model.fill_placeholders(
          inputs, labels, is_training_phase = True))

    self.writer.add_summary(summary, step_number)
    return train_result

  def train_unsupervised_batch(self, inputs, step_number):
    train_result, summary = self.session.run(
        [self.model.unsupervised_train_step, self.model.unsupervised_summaries],
        self.model.fill_placeholders(inputs, is_training_phase = True))

    self.writer.add_summary(summary, step_number)
    return train_result

  def test(self, inputs, labels, step_number):
    accuracy, summary = self.session.run(
        [self.model.accuracy_measure, self.model.test_summaries],
        self.model.fill_placeholders(inputs, labels, is_training_phase = False))

    self.writer.add_summary(summary, step_number)
    return accuracy
