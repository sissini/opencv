import tensorflow as tf 
mnist = tf.keras.datasets.mnist

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models
from keras.models import load_model

if __name__ == "__main__":

	#导入mnist  (import mnist)
	(x_train, y_train), (x_test, y_test) = mnist.load_data()
	## image size
	img_row, img_col = 28, 28

	## 0-1 transform
	x_train, x_test = x_train/255.0, x_test/255.0
	## one-hot
	y_train = keras.utils.to_categorical(y_train, num_classes=10)
	y_test = keras.utils.to_categorical(y_test, num_classes=10)

	##reshap image (black-white image --> 1)
	x_train = x_train.reshape(x_train.shape[0], img_row, img_col, 1)
	x_test = x_test.reshape(x_test.shape[0], img_row, img_col, 1)

	## lenet-5 model
	model = models.Sequential()

	model.add(layers.Conv2D(32, kernel_size=(5,5), activation='relu', input_shape=(img_row,img_col,1)))
	model.add(layers.MaxPooling2D(pool_size=(2,2)))
	model.add(layers.Conv2D(64, (5,5), activation = 'relu'))
	model.add(layers.MaxPooling2D(pool_size=(2,2)))
	#model.add(layers.Conv2D(128,()))
	model.add(layers.Flatten())
	model.add(layers.Dense(500, activation='relu'))
	model.add(layers.Dense(10, activation='softmax'))

	#loss
	model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.SGD(), metrics=['accuracy'])

	#train start
	model.fit(x_train, y_train, batch_size=128, epochs=10, verbose = 2, validation_data= (x_test,y_test))

	#loss and accuracy
	score = model.evaluate(x_test,y_test)
	print('test loss:',score[0])
	print('test accuracy:', score[1])
	#save trained data in file h5
	model.save('my_mnist_first.h5')

	# # 保存
	# json_string = model.to_jsn()
	# open('/Users/apple1/desktop/python_work/keras/mnist/my_model_architecture.json', 'w').write(json_string)

	# model.save_weights('/Users/apple1/desktop/python_work/keras/mnist/my_model_weight.h5')
