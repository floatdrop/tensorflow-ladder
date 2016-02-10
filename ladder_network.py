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
    return self._run(self.model.supervised_train_step,
        summary_action = self.model.supervised_summaries,
        step_number = step_number,
        inputs = inputs,
        labels = labels,
        is_training_phase = True)

  def train_unsupervised_batch(self, inputs, step_number):
    return self._run(self.model.unsupervised_train_step,
        summary_action = self.model.unsupervised_summaries,
        step_number = step_number,
        inputs = inputs,
        is_training_phase = True)

  def test(self, inputs, labels, step_number):
    return self._run(self.model.accuracy_measure,
        summary_action = self.model.test_summaries,
        step_number = step_number,
        inputs = inputs,
        labels = labels,
        is_training_phase = False)

  def _run(self, action, summary_action, step_number, inputs, labels = None, is_training_phase = True):
    variable_placements = self.model.placeholders.placements(
        inputs, labels, is_training_phase)
    action_result, summary = self.session.run(
        [action, summary_action], variable_placements)
    self.writer.add_summary(summary, step_number)
    return action_result


class Model:
  def __init__(self, input_layer_size, class_count):
    self.hyperparameters = {
      "learning_rate": 0.003,
      "noise_level": 0.2,
      "denoising_cost_multiplier": 0.00001,
      "encoder_layer_definitions": [
        (100, tf.nn.relu),
        (50, tf.nn.relu),
        (class_count, tf.nn.softmax)
      ]
    }

    self.placeholders = _Placeholders(input_layer_size, class_count)
    self.output = _ForwardPass(self.placeholders, self.hyperparameters)
    self.accuracy_measure = self._accuracy_measure(
        self.placeholders, self.output)
    self.supervised_train_step = self._supervised_train_step(
        self.placeholders, self.output)
    self.unsupervised_train_step = self._unsupervised_train_step(
        self.placeholders, self.output)

    self.unsupervised_summaries = tf.merge_all_summaries("unsupervised")
    self.supervised_summaries = tf.merge_all_summaries("supervised")
    self.test_summaries = tf.merge_all_summaries("test")

  def _accuracy_measure(self, placeholders, output):
    with tf.name_scope("accuracy_measure") as scope:
      correct_prediction = tf.equal(tf.argmax(output.clean_label_probabilities, 1), tf.argmax(placeholders.labels, 1))
      accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
      tf.scalar_summary("test accuracy", accuracy, ["test"])
      return accuracy

  def _supervised_train_step(self, placeholders, output):
    with tf.name_scope("supervised_training") as scope:
      total_cost = self._total_cost(placeholders, output,
          self.hyperparameters["denoising_cost_multiplier"])
      return self._optimizer(self.hyperparameters["learning_rate"], total_cost)

  def _unsupervised_train_step(self, placeholders, output):
    with tf.name_scope("unsupervised_training") as scope:
      denoising_cost = self._total_denoising_cost(
          placeholders, output,
          self.hyperparameters["denoising_cost_multiplier"], ["unsupervised"])
      return self._optimizer(self.hyperparameters["learning_rate"], denoising_cost)

  def _optimizer(self, learning_rate, cost_function):
    with tf.name_scope("optimizer") as scope:
      optimizer = tf.train.AdamOptimizer(learning_rate)
      return optimizer.minimize(cost_function)

  def _total_cost(self, placeholders, output, denoising_cost_multiplier):
    with tf.name_scope("total_cost") as scope:
      cross_entropy = self._cross_entropy(placeholders, output)
      denoising_cost = self._total_denoising_cost(
          placeholders, output, denoising_cost_multiplier, ["supervised"])
      total_cost = cross_entropy + denoising_cost
      tf.scalar_summary("total cost", cross_entropy, ["supervised"])
      return total_cost

  def _cross_entropy(self, placeholders, output):
    with tf.name_scope("cross_entropy_cost") as scope:
      cross_entropy = -tf.reduce_mean(
          placeholders.labels * tf.log(output.corrupted_label_probabilities))
      tf.scalar_summary("cross entropy", cross_entropy, ["supervised"])
      return cross_entropy

  def _total_denoising_cost(self, placeholders, output, cost_multiplier, summary_tags):
    with tf.name_scope("denoising_cost") as scope:
      layer_costs = [self._layer_denoising_cost(encoder, decoder, cost_multiplier)
        for (encoder, decoder)
        in zip(output.clean_encoder_outputs, reversed(output.decoder_outputs))]
      denoising_cost = sum(layer_costs)

      for index, layer_cost in enumerate(layer_costs):
        tf.scalar_summary("layer %i denoising cost" % index, layer_cost, summary_tags)
      tf.scalar_summary("total denoising cost", denoising_cost, summary_tags)

      return denoising_cost

  def _layer_denoising_cost(self, encoder, decoder, cost_multiplier):
    return self._mean_squared_error(
      encoder.pre_activation, decoder.post_2nd_normalization) * cost_multiplier

  def _mean_squared_error(self, expected, actual):
    return tf.reduce_mean(tf.pow(expected - actual, 2))

class _Placeholders:
  def __init__(self, input_layer_size, class_count):
    with tf.name_scope("placeholders") as scope:
      self.inputs = tf.placeholder(tf.float32, [None, input_layer_size], name = 'inputs')
      self.labels = tf.placeholder(tf.float32, [None, class_count], name = 'labels')
      self.is_training_phase = tf.placeholder(tf.bool, name = 'is_training_phase')

  def placements(self, inputs, labels = None, is_training_phase = True):
    if labels is None:
      labels = numpy.zeros([inputs.shape[0], _layer_size(self.labels)])
    return {
      self.inputs: inputs,
      self.labels: labels,
      self.is_training_phase: is_training_phase
    }


class _ForwardPass:
  def __init__(self, placeholders, hyperparameters):
    encoder_layer_definitions = hyperparameters["encoder_layer_definitions"]
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
    # FIXME: Actually the first decoder layer shold get the correct label from above
    with tf.name_scope("decoder") as scope:
      encoder_layers = reversed(zip(clean_encoder_layers, corrupted_encoder_layers))
      layer_accumulator = [None]
      for clean_layer, corrupted_layer in encoder_layers:
        layer = _DecoderLayer(
            clean_encoder_layer = clean_layer,
            corrupted_encoder_layer = corrupted_layer,
            previous_decoder_layer = layer_accumulator[-1],
            is_training_phase = is_training_phase)
        layer_accumulator.append(layer)
      return layer_accumulator[1:]


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
      clean_encoder_layer, corrupted_encoder_layer,
      previous_decoder_layer = None, is_training_phase = True):
    with tf.name_scope("decoder_layer") as scope:
      is_first_decoder_layer = previous_decoder_layer is None
      if is_first_decoder_layer:
        pre_1st_normalization = corrupted_encoder_layer.post_activation
      else:
        input_size = _layer_size(previous_decoder_layer.post_denoising)
        output_size = _layer_size(clean_encoder_layer.post_activation)
        weights = _weight_variable([input_size, output_size])
        pre_1st_normalization = tf.matmul(
          previous_decoder_layer.post_denoising, weights)

      pre_denoising, _, _ = batch_norm(pre_1st_normalization, is_training_phase = is_training_phase)
      post_denoising = self._denoise(
        corrupted_encoder_layer.pre_activation, pre_denoising)
      post_2nd_normalization = \
        (post_denoising - clean_encoder_layer.batch_mean) / clean_encoder_layer.batch_std

      self.post_denoising = post_denoising
      self.post_2nd_normalization = post_2nd_normalization

  def _denoise(self, from_left, from_above):
    mu = self._modulate(from_above)
    v = self._modulate(from_above)
    return (from_left - mu) * v + mu

  def _modulate(self, u):
    a = [_weight_variable([_layer_size(u)]) for i in xrange(5)]
    return a[0] * tf.nn.sigmoid(a[1] * u + a[2]) + a[3] * u + a[4]


def _weight_variable(shape):
  with tf.name_scope("weight") as scope:
    initial = tf.truncated_normal(shape, stddev = 0.1)
    return tf.Variable(initial)

def _layer_size(layer_output):
  return layer_output.get_shape()[-1].value


