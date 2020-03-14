import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf 
from PIL import Image
import cv2

#load the data from file h5
model = tf.keras.models.load_model('my_mnist_first.h5')

def pre_pic(picName):
    # 先打开传入的原始图片  open the picture  
    img = Image.open(picName)
    # 使用消除锯齿的方法resize图片  resize the picture
    reIm = img.resize((28,28),Image.ANTIALIAS)
    # 变成灰度图，转换成矩阵	transform picture to Gray and save as a array
    im_arr = np.array(reIm.convert("L"))
    return im_arr


im = pre_pic('.png')
im1 = 255 - im
print(im1.shape)
print('输入数字：') # the input number is: 

# # im = Image.open('1.png')
# im.show()
# im1.show()

im1 = im1.reshape(1, 28, 28, 1)
im1 = im1.astype('float32')/255

predict = model.predict_classes(im1)
print ('识别为：') # detect as :
print (predict)
