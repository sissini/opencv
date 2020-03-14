import tensorflow as tf 
import matplotlib.pyplot as plt
import numpy as np 
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

train_images = mnist.train.images
train_labels = mnist.train.labels

# validation_images = mnist.validation.images
# validation_labels = mnist.validation.labels

# test_images = mnist.test.images
# test_labels = mnist.test.labels


# fig, ax = plt.subplots(
#     nrows=5,
#     ncols=5,
#     sharex='all',
#     sharey='all', )

# ax = ax.flatten()
# for i in range(25):
#     img = mnist.train.images[i].reshape(28, 28)
#     ax[i].imshow(img, cmap='Greys', interpolation='nearest')

# ax[0].set_xticks([])
# ax[0].set_yticks([])
# plt.tight_layout()
# plt.show()

X = []
Y = []

for i in range(10):
    x = i
    y = np.sum(train_labels == i)
    X.append(x)
    Y.append(y)
    plt.text(x, y, '%s' % y, ha='center', va= 'bottom')

plt.bar(X, Y, facecolor='#9999ff', edgecolor='white')
plt.xticks(X)
plt.show()