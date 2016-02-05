import random
import input_data
import ladder_network

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

model = ladder_network.Model(
  input_layer_size = 784,
  class_count = 10
)

with ladder_network.Session(model) as session:
  for i in range(100):
    for j in range(100):
      images, labels = mnist.train.next_batch(100)
      session.train_supervised_batch(
        images,
        labels,
        step_number = 2 * (i * 100 + j)
      )

      images, _ = mnist.train_unlabeled.next_batch(100)
      session.train_unsupervised_batch(
        images,
        step_number = 2 * (i * 100 + j) + 1
      )

    print session.test(mnist.test.images, mnist.test.labels)
