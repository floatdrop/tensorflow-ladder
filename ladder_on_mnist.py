import random
import input_data
import ladder_network

LABELED_EXAMPLES = 1000

def partition_indices(total, first_list_size):
  all_indices = xrange(total)
  first_list = random.sample(all_indices, first_list_size)
  second_list = list(set(all_indices) - set(first_list))
  return first_list, second_list

def reshape_to_4d(array):
  return array.reshape(array.shape + (1, 1))

def add_unlabeled_split(data_sets, label_count):
  labeled_indices, unlabeled_indices = partition_indices(data_sets.train.num_examples, LABELED_EXAMPLES)
  labeled_train_images = data_sets.train.images[labeled_indices]
  labeled_train_labels = data_sets.train.labels[labeled_indices]
  unlabeled_train_images = data_sets.train.images[unlabeled_indices]
  unlabeled_train_labels = data_sets.train.labels[unlabeled_indices]
  unlabeled_train_labels.fill(0)

  labeled_train_images = reshape_to_4d(labeled_train_images)
  unlabeled_train_images = reshape_to_4d(unlabeled_train_images)

  data_sets.train_labeled = input_data.DataSet(labeled_train_images, labeled_train_labels)
  data_sets.train_unlableled = input_data.DataSet(unlabeled_train_images, unlabeled_train_labels)


mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
add_unlabeled_split(mnist, LABELED_EXAMPLES)

print len(mnist.train_labeled.images), len(mnist.train_labeled.images)

model = ladder_network.Model(
  input_layer_size = 784,
  class_count = 10
)

with ladder_network.Session(model) as session:
  for i in range(100):
    for j in range(100):
      images, labels = mnist.train_labeled.next_batch(100)
      session.train_batch(images, labels, i * 100 + j)

    print session.test(mnist.test.images, mnist.test.labels)
