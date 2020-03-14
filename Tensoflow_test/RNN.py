import tensorflow as tf 
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("./data", one_hot = True)

learning_rate = 0.01
steps = 10000
batch_size = 128

n_inputs = 28
n_steps = 28
n_hidden_units = 128
n_class = 10

x = tf.placeholder(tf.float32,[None,n_steps,n_inputs])
y = tf.placeholder(tf.float32),[None, n_class]

weights = {'in': tf.Variable(tf.random_normal([n_inputs,n_hidden_units])),
			'out': tf.Variable(tf.random_normal([n_hidden_units,n_class]))}

biases = {'in': tf.Variable(tf.constant(0.1), shape=([n_hidden_units,]), 
		'out': tf.Variable(tf.constant(0.1), shape = ([n_class,])}


def rnn(input, weights, biases):
 	input = tf.reshape(input,[-1,n_inputs])
	ws_in = tf.matmul(input,weights['in'])+ biases['in']
	ws_in = tf.reshape(ws_in,[-1,n_steps, n_hidden_units])

	lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden_units, forget_bias = 1.0, state_is_Tuple = True)
	_init_state = lstm_cell.zero_state(batch_size, dtype = tf.float32)
	outputs, states = tf.nn.dymanic_rnn(lstm_cell, X_in, initial_state = _init_state, time_major =False)

	results = tf.matmul(states[1], weights['out'])+ biases['out']

	return results

prediction = rnn(x,weights,biases)
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(prediction, y))
train = tf.train.AdamOptimizer(learning_rate).minimize(loss)

correct_pre = tf.equal(tf.argmax(prediction,1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pre, tf.float32))

init = tf.initialize_all_variables
with tf.Session as sess:
	sess.run(init)
	for i in range(steps):
		batch_xs, batch_ys = mnist.train.next_batch(batch_size)
		batch_xs = batch_xs.reshape([batch_size, n_steps, n_inputs])
		sess.run(train, feed_dict={x: batch_xs, y: batch_ys})
		if i%50 ==0:
			print(sess.run(accurancy, feed_dict={x: batch_xs, y: batch_ys}))

