import tensorflow as tf 

state = tf.Variable(0,name='counter')
one = tf.constant(2)
new_value = tf.add(state,one)
update = tf.assign(state, new_value)

init = tf.initialize_all_variables()

with tf.Session() as sess:
	sess.run(init)
	for _ in range(3):
		new_number = sess.run(update)
		print(new_number)
