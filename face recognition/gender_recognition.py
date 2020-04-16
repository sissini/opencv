import cv2
import numpy as np 
from keras.models import load_model

img = cv2.imread('/Users/apple1/Desktop/编程/python/opencv/faceRecognition/images/lidr.jpeg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

face_classifier = cv2.CascadeClassifier('/Users/apple1/Desktop/编程/python/opencv/books/OpenCV 3计算机视觉 Python实现2/pycv-master/chapter5/cascades/haarcascade_frontalface_default.xml')
faces = face_classifier.detectMultiScale(gray, scaleFactor= 1.2, minSize = (140, 140))

gender_classifier = load_model('/Users/apple1/Desktop/编程/python/opencv/faceRecognition/simple_CNN.81-0.96.hdf5')
gender_labels = {0: 'female', 1:'male'}
color = (0,0,255)

for (x,y,w,h) in faces:
	face = img[(y-60):(y+h+60), (x-30):(x+w+30)]
	face = cv2.resize(face, (48,48))
	face = np.expand_dims(face,0)
	face = face/255.0
	gender_label_arg = np.argmax(gender_classifier.predict(face))
	gender = gender_labels[gender_label_arg]
	cv2.rectangle(img, (x,y), (x+w, y+h), color, 2)
	img = cv2.putText(img, gender, (x+w, y),cv2.FONT_HERSHEY_DUPLEX,1.0, color, 3)

cv2.imshow('image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
