import tensorflow as tf
import numpy
from tensorflow.python import control_flow_ops
from batch_normalization import batch_norm
from time import strftime



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


class Model:
  def __init__(self, input_layer_size, class_count):
    self.hyperparameters = {
      "learning_rate": 0.001,
      "cross_entropy_training_weight": 100,
      "noise_level": 0.2
    }

    self.placeholders = self._data_placeholders(
        input_layer_size, class_count)
    self.output = _ForwardPass(self.placeholders, self.hyperparameters)
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
      labels = numpy.zeros([inputs.shape[0], _layer_size(self.placeholders.labels)])
    return {
      self.placeholders.inputs: inputs,
      self.placeholders.labels: labels,
      self.placeholders.is_training_phase: is_training_phase
    }

  def _data_placeholders(self, input_layer_size, class_count):
    with tf.name_scope("placeholders") as scope:
      placeholders = _Record()
      placeholders.inputs = tf.placeholder(tf.float32, [None, input_layer_size], name = 'inputs')
      placeholders.labels = tf.placeholder(tf.float32, [None, class_count], name = 'labels')
      placeholders.is_training_phase = tf.placeholder(tf.bool, name = 'is_training_phase')
      return placeholders

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
      correct_prediction = tf.equal(tf.argmax(output.clean_label_probabilities, 1), tf.argmax(placeholders.labels, 1))
      accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
      tf.scalar_summary("test accuracy", accuracy, ["test"])
      return accuracy

  def _optimizer(self, learning_rate, cost_function):
    with tf.name_scope("optimizer") as scope:
      optimizer = tf.train.AdamOptimizer(learning_rate)
      return optimizer.minimize(cost_function)

  def _total_cost(self, placeholders, output, cross_entropy_training_weight):
    with tf.name_scope("total_cost") as scope:
      cross_entropy = self._cost_entropy(placeholders, output)
      autoencoder_cost = self._autoencoder_cost(placeholders, output, "supervised")
      return cross_entropy_training_weight * cross_entropy + autoencoder_cost

  def _autoencoder_cost(self, placeholders, output, summary_tag):
    with tf.name_scope("autoencoder_cost") as scope:
      clean_encoder_outputs = [encoder.pre_activation
        for encoder in output.clean_encoder_outputs]
      decoder_outputs = [decoder.post_2nd_normalization
        for decoder in list(reversed(output.decoder_outputs))]

      assert all(encoder.get_shape().is_compatible_with(decoder.get_shape())
        for (encoder, decoder) in zip(clean_encoder_outputs, decoder_outputs))

      layer_costs = [tf.reduce_mean(tf.pow(encoder - decoder, 2))
        for (encoder, decoder) in zip(clean_encoder_outputs, decoder_outputs)]

      for index, layer_cost in enumerate(layer_costs):
        tf.scalar_summary("layer %i autoencoder cost" % index, layer_cost, [summary_tag])

      autoencoder_cost = sum(layer_costs)
      tf.scalar_summary("autoencoder cost", autoencoder_cost, [summary_tag])
      return autoencoder_cost

  def _cost_entropy(self, placeholders, output):
    with tf.name_scope("cross_entropy_cost") as scope:
      cross_entropy = -tf.reduce_mean(
          placeholders.labels * tf.log(output.corrupted_label_probabilities))
      tf.scalar_summary("cross entropy", cross_entropy, ["supervised"])
      return cross_entropy


class _ForwardPass:
  def __init__(self, placeholders, hyperparameters):
    encoder_layer_definitions = [
      (100, tf.nn.relu),
      (50, tf.nn.relu),
      (_layer_size(placeholders.labels), tf.nn.softmax)
    ]
    clean_encoder_outputs = self._encoder_layers(
        input_layer = placeholders.inputs,
        other_layer_definitions = encoder_layer_definitions,
        noise_level = 0.0,
        is_training_phase = placeholders.is_training_phase)

    corrupted_encoder_outputs = self._encoder_layers(
        input_layer = placeholders.inputs,
        other_layer_definitions = encoder_layer_definitions,
        noise_level = hyperparameters["noise_level"],
        is_training_phase = placeholders.is_training_phase)

    decoder_outputs = self._decoder_layers(
        clean_encoder_layers = clean_encoder_outputs,
        corrupted_encoder_layers = corrupted_encoder_outputs,
        is_training_phase = placeholders.is_training_phase)

    self.clean_label_probabilities = clean_encoder_outputs[-1].post_activation
    self.corrupted_label_probabilities = corrupted_encoder_outputs[-1].post_activation
    self.autoencoded_inputs = decoder_outputs[-1]
    self.clean_encoder_outputs = clean_encoder_outputs
    self.corrupted_encoder_outputs = corrupted_encoder_outputs
    self.decoder_outputs = decoder_outputs

  def _encoder_layers(self,
      input_layer, other_layer_definitions,
      noise_level, is_training_phase):
    with tf.name_scope("encoder") as scope:
      first_encoder_layer = _InputLayerWrapper(input_layer)

      layer_accumulator = [first_encoder_layer]
      for (layer_size, non_linearity) in other_layer_definitions:
        layer_output = _EncoderLayer(
            inputs = layer_accumulator[-1].post_activation,
            output_size = layer_size,
            non_linearity = non_linearity,
            noise_level = noise_level,
            is_training_phase = is_training_phase)

        layer_accumulator.append(layer_output)
      return layer_accumulator

  def _decoder_layers(self, clean_encoder_layers, corrupted_encoder_layers,
        is_training_phase):
    with tf.name_scope("decoder") as scope:
      last_corrupted_encoder_layer = corrupted_encoder_layers[-1]
      last_clean_encoder_layer = clean_encoder_layers[-1]
      rest_of_encoder_layers = zip(clean_encoder_layers, corrupted_encoder_layers)[:-1]

      first_decoder_layer = _DecoderLayer(
          previous_decoder_layer = last_corrupted_encoder_layer,
          corrupted_encoder_layer = last_corrupted_encoder_layer,
          clean_encoder_layer = last_clean_encoder_layer,
          is_training_phase = is_training_phase,
          is_first_decoder = True
      )

      layer_accumulator = [first_decoder_layer]
      for clean_layer, corrupted_layer in reversed(rest_of_encoder_layers):
        layer = _DecoderLayer(
            previous_decoder_layer = layer_accumulator[-1],
            clean_encoder_layer = clean_layer,
            corrupted_encoder_layer = corrupted_layer,
            is_training_phase = is_training_phase)
        layer_accumulator.append(layer)
      return layer_accumulator


class _InputLayerWrapper:
  def __init__(self, input_layer):
    self.pre_activation = input_layer
    self.post_activation = input_layer
    self.batch_mean = tf.zeros_like(input_layer)
    self.batch_std = tf.ones_like(input_layer)


class _EncoderLayer:
  def __init__(self,
      inputs, output_size, non_linearity,
      noise_level, is_training_phase):
    with tf.name_scope("encoder_layer") as scope:
      weights = _weight_variable([_layer_size(inputs), output_size])
      self.pre_normalization = tf.matmul(inputs, weights)
      pre_noise, self.batch_mean, self.batch_std = batch_norm(
          self.pre_normalization, is_training_phase = is_training_phase)
      self.pre_activation = pre_noise + tf.random_normal(
          [output_size], mean = 0.0, stddev = noise_level)
      self.post_activation = non_linearity(self._beta_gamma(self.pre_activation))

  def _beta_gamma(self, inputs):
    layer_size = _layer_size(inputs);
    beta = tf.Variable(tf.constant(0.0,
        shape = [layer_size]), name = 'beta', trainable = True)
    gamma = tf.Variable(tf.constant(1.0,
        shape = [layer_size]), name = 'gamma', trainable = True)
    return gamma * (inputs + beta)


class _DecoderLayer:
  def __init__(self,
      previous_decoder_layer, clean_encoder_layer, corrupted_encoder_layer,
      is_training_phase, is_first_decoder = False):
    with tf.name_scope("decoder_layer") as scope:
      input_size = _layer_size(previous_decoder_layer.post_activation)
      output_size = _layer_size(clean_encoder_layer.post_activation)

      if is_first_decoder:
        pre_1st_normalization = previous_decoder_layer.post_activation
      else:
        weights = _weight_variable([input_size, output_size])
        pre_1st_normalization = tf.matmul(
          previous_decoder_layer.post_activation, weights)

      pre_activation, _, _ = batch_norm(pre_1st_normalization, is_training_phase = is_training_phase)
      post_activation = tf.nn.relu(pre_activation)

      post_2nd_normalization = \
        (post_activation - clean_encoder_layer.batch_mean) / clean_encoder_layer.batch_std

      self.pre_activation = pre_activation
      self.post_activation = post_activation
      self.post_2nd_normalization = post_2nd_normalization


class _Record:
  pass

def _weight_variable(shape):
  with tf.name_scope("weight") as scope:
    initial = tf.truncated_normal(shape, stddev = 0.1)
    return tf.Variable(initial)

def _layer_size(layer_output):
  return layer_output.get_shape()[-1].value


