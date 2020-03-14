import tensorflow as tf
import numpy as np 
import matplotlib.pyplot as plt 

def nn(input,in_size,out_size,activation_function = None):
	weights = tf.Variable(tf.random_normal([in_size,out_size]))
	bias = tf.Variable(tf.zeros([1,out_size])+0.1)
	wx = tf.matmul(input,weights)+bias
	if activation_function == None:
		output = wx
	else:
		output = activation_function(wx)
	return output


# rng = np.random.RandomState(0)
# x_data = rng.rand(5,1)
x_data = np.linspace(-1,1,300)[:, np.newaxis]
y_data = np.square(x_data)


x = tf.placeholder(tf.float32,[None,1])
y = tf.placeholder(tf.float32,[None,1])

l1 = nn(x,1,10, activation_function= tf.nn.relu)
y_ = nn(l1,10,1)

loss = tf.reduce_mean(tf.square(y_-y))
train = tf.train.GradientDescentOptimizer(0.2).minimize(loss)

init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.scatter(x_data,y_data)
plt.ion()
plt.show()


for i in range(1000):
	sess.run(train, feed_dict = {x:x_data, y:y_data})
	if i % 20 == 0:
		print(sess.run(loss, feed_dict = {x:x_data, y:y_data}))
		try:
			ax.lines.remove(lines[0])
		except Exception: pass
		prediction = sess.run(y_, feed_dict={x: x_data})
		lines = ax.plot(x_data, prediction, 'r', lw=5)
		plt.pause(0.1)

