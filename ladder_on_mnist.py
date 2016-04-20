import random
import input_data
import ladder_network

mnist = input_data.read_data_sets(
    "MNIST_data/",
    one_hot=True,
    labeled_size=5000,
    validation_size=5000)

print mnist.train_unlabeled.num_examples, "unlabeled training examples"
print mnist.train_labeled.num_examples, "labeled training examples"
print mnist.validation.num_examples, "validation examples"
print mnist.test.num_examples, "test examples"

model = ladder_network.Model(input_layer_size = 784, class_count = 10)

with ladder_network.Session(model) as session:
  for step in xrange(1000):
    if step % 5 == 0:
      images, labels = mnist.train_labeled.next_batch(100)
      session.train_supervised_batch(images, labels, step)
    else:
      images, _ = mnist.train_unlabeled.next_batch(100)
      session.train_unsupervised_batch(images, step)

    if step % 200 == 0:
      print session.test(
        mnist.validation.images, mnist.validation.labels, step)
