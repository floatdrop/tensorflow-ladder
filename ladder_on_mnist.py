import input_data
import ladder_network

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

model = ladder_network.Model(
  input_dimension_count = 784,
  label_count = 10
)

with ladder_network.Session(model) as session:
  for i in range(100):
    session.train_batch(*mnist.train.next_batch(100))

  print session.test(mnist.test.images, mnist.test.labels)
  print session.sample(mnist.test.images[0:10])
