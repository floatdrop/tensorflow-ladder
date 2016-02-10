import tensorflow as tf
import numpy
from tensorflow.python import control_flow_ops
from batch_normalization import batch_norm
from time import strftime

class Model:
  def __init__(self, input_layer_size, class_count):
    self.hyperparameters = {
      "learning_rate": 0.001,
      "cross_entropy_training_weight": 100,
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
    clean_encoder_outputs = self._encoder_layers(
        input_layer = placeholders.inputs,
        other_layer_definitions = encoder_layer_definitions,
        noise_level = 0.0,
        is_training_phase = placeholders.is_training_phase)

    corrupted_encoder_outputs = self._encoder_layers(
        input_layer = placeholders.inputs,
        other_layer_definitions = encoder_layer_definitions,
        noise_level = self.hyperparameters["noise_level"],
        is_training_phase = placeholders.is_training_phase)

    decoder_outputs = self._decoder_layers(
        clean_encoder_layers = clean_encoder_outputs,
        corrupted_encoder_layers = corrupted_encoder_outputs,
        is_training_phase = placeholders.is_training_phase)

    output = self._Record()
    output.clean_label_probabilities = clean_encoder_outputs[-1].post_activation
    output.corrupted_label_probabilities = corrupted_encoder_outputs[-1].post_activation
    output.autoencoded_inputs = decoder_outputs[-1]
    output.clean_encoder_outputs = clean_encoder_outputs
    output.corrupted_encoder_outputs = corrupted_encoder_outputs
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
      correct_prediction = tf.equal(tf.argmax(output.clean_label_probabilities, 1), tf.argmax(placeholders.labels, 1))
      accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
      tf.scalar_summary("test accuracy", accuracy, ["test"])
      return accuracy

  def _encoder_layers(self,
      input_layer, other_layer_definitions,
      noise_level, is_training_phase):
    with tf.name_scope("encoder") as scope:
      input_layer_holder = self._Record()
      input_layer_holder.post_activation = input_layer
      input_layer_holder.pre_activation = input_layer
      input_layer_holder.batch_mean = 0.0 # ???
      input_layer_holder.batch_std = 1.0 # ???

      layer_outputs = [input_layer_holder]
      for (layer_size, non_linearity) in other_layer_definitions:
        layer_output = self._fully_connected_layer(
            inputs = layer_outputs[-1].post_activation,
            output_size = layer_size,
            non_linearity = non_linearity,
            noise_level = noise_level,
            is_training_phase = is_training_phase)

        layer_outputs.append(layer_output)
      return layer_outputs

  def _decoder_layers(self, clean_encoder_layers, corrupted_encoder_layers,
        is_training_phase):
    with tf.name_scope("decoder") as scope:
      decoder_layers = [corrupted_encoder_layers[-1]]
      encoder_layers = zip(clean_encoder_layers, corrupted_encoder_layers)[:-1]
      for clean, corrupted in reversed(encoder_layers):
        layer_output = self._decoder_layer(
            previous_decoder_layer = decoder_layers[-1],
            clean_encoder_layer = clean,
            corrupted_encoder_layer = corrupted,
            is_training_phase = is_training_phase)
        decoder_layers.append(layer_output)
      return decoder_layers

  def _fully_connected_layer(self,
      inputs, output_size, non_linearity,
      noise_level, is_training_phase):
    with tf.name_scope("layer") as scope:
      weights = self._weight_variable([self._layer_size(inputs), output_size])
      pre_normalization = tf.matmul(inputs, weights)
      pre_noise, batch_mean, batch_std = batch_norm(pre_normalization, is_training_phase = is_training_phase)
      pre_activation = pre_noise + tf.random_normal([output_size],
          mean = 0.0, stddev = noise_level)
      post_activation = non_linearity(self._beta_gamma(pre_activation))

      layer_output = self._Record()
      layer_output.pre_normalization = pre_normalization
      layer_output.pre_activation = pre_activation
      layer_output.post_activation = post_activation
      layer_output.batch_mean = batch_mean
      layer_output.batch_std = batch_std
      return layer_output

  def _decoder_layer(self,
      previous_decoder_layer, clean_encoder_layer, corrupted_encoder_layer,
      is_training_phase):
    with tf.name_scope("decoder_layer") as scope:
      inputs = previous_decoder_layer.post_activation
      output_size = self._layer_size(clean_encoder_layer.post_activation)
      weights = self._weight_variable([self._layer_size(inputs), output_size])
      pre_1st_normalization = tf.matmul(inputs, weights)
      pre_activation, _, _ = batch_norm(pre_1st_normalization, is_training_phase = is_training_phase)
      post_activation = tf.nn.relu(pre_activation)

      post_2nd_normalization = \
        (post_activation - clean_encoder_layer.batch_mean) / clean_encoder_layer.batch_std

      layer_output = self._Record()
      layer_output.pre_activation = pre_activation
      layer_output.post_activation = post_activation
      layer_output.post_2nd_normalization = post_2nd_normalization
      return layer_output    

  def _beta_gamma(self, inputs):
    layer_size = self._layer_size(inputs);
    beta = tf.Variable(tf.constant(0.0,
        shape = [layer_size]), name = 'beta', trainable = True)
    gamma = tf.Variable(tf.constant(1.0,
        shape = [layer_size]), name = 'gamma', trainable = True)
    return gamma * (inputs + beta)


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
      decoder_outputs = [decoder.pre_activation
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
