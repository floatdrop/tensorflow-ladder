import random
import input_data
import ladder_network

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

model = ladder_network.Model(
  input_layer_size = 784,
  class_count = 10
)

with ladder_network.Session(model) as session:
  for step in xrange(10000000):
    if step % 2 == 0:
      images, labels = mnist.train.next_batch(100)
      session.train_supervised_batch(images, labels, step)
    else:
      images, _ = mnist.train_unlabeled.next_batch(100)
      session.train_unsupervised_batch(images, step)

    if step % 200 == 0:
      print session.test(
        mnist.validation.images, mnist.validation.labels, step)
