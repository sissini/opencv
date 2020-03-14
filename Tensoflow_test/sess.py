import tensorflow as tf 

matrix1 = tf.constant([[3,4]])
matrix2 = tf.constant([[2], [5]])
product = tf.matmul(matrix1,matrix2)

# methode 1 
with tf.Session() as sess:
	result = sess.run(product) 
	print(result)

# # methode 2
# sess = tf.Session()
# result2 = sess.run(product)
# print(result2)