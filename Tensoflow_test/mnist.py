import tensorflow as tf 
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("./data/", one_hot = True)

learning_rate = 0.001
num_steps = 500
batch_size = 128
display_step = 10

num_input = 784 # 28 * 28
num_classes = 10
dropout = 0.75

X = tf.placeholder(tf.float32, [None, num_input])
Y = tf.placeholder(tf.float32, [None, num_classes])
keep_prob = tf.placeholder(tf.float32)

def conv2d (x,w,b,strides = 1):
	x = tf.nn.conv2d(x,w,strides = [1, strides, strides, 1], padding = 'SAME')
	x = tf.nn.bias_add(x,b)
	return tf.nn.relu(x)

def maxpool2d(x, k=2):
	return tf.nn.max_pool(x, ksize=[1,k,k,1], strides=[1,k,k,1], padding = 'SAME')


def conv_net(x, weights, biases, dropout):
	x = tf.reshape(x, shape=[-1,28,28,1])
	conv1 = conv2d(x, weights['wc1'], biases['bc1'])
	conv1 = maxpool2d(conv1, k=2)

	conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
	conv2 = maxpool2d(conv2, k=2)

	fc1 = tf.reshape(conv2, [-1, weights['wd1'].get_shape().as_list()[0]])
	fc1 = tf.add(tf.matmul(fc1, weights['wd1'], biases['bd1']))
	fc1 = tf.nn.relu(fc1)
	fc1 = tf.nn.dropout(fc1, dropout)
	output = tf.add(tf.matmul(fc1, weights['output']), biases['output'])
	return output


weights = {'wc1': tf.Variable(tf.random_normal([5,5,1,32])),
			'wc2': tf.Variable(tf.random_normal([5,5,32,64])),
			'wd1': tf.Variable(tf.random_normal([7*7*64, 1024])),
			'wd2': tf.Variable(tf.random_normal([1024, num_classes]))}

biases = {'bc1': tf.Variable(tf.random_normal([32])),
			'bc2': tf.Variable(tf.random_normal([64])),
			'bd1': tf.Variable(tf.random_normal([1024])),
			'bd2': tf.Variable(tf.random_normal([num_classes]))}


logits = conv_net(X, weights, biases, keep_prob)
prediction = tf.nn.softmax(logits)

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))
train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)


init = tf.global_variable_inizializer()

with tf.Session() as sess:
	sess.run(init)
	for step in range(1, num_steps):
		batch_x, batch_y = mnist.train.next_batch(batch_size)
		sess.run(train_op, feed_dict = {X: batch_x, Y: batch_y, keep_prob: dropout})
		if step % display_step == 0 or step == 1:
			print("step"+ sess.run(loss, feed_dict = {X: batch_x, Y: batch_y, keep_prob: dropout}))
	print("optimization finished!")
