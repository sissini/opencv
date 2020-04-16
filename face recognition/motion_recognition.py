import cv2
import numpy as np 
import datetime
from keras.models import load_model

startTime = datetime.datetime.now()
emotion_classifier = load_model('/Users/apple1/Desktop/编程/python/opencv/faceRecognition/simple_CNN.81-0.96.hdf5')

endTime = datetime.datetime.now()
print(endTime - startTime)

emotion_labels = {	0: 'angry', 
				    1: 'disgust',
				    2: 'fear',
				    3: 'happy',
				    4: 'sad',
				    5: 'superise',
				    6: 'calm'	}
img =cv2.imread('/Users/apple1/Desktop/编程/python/Tesseract_ORC/reisen.jpg')
face_classifier = cv2.CascadeClassifier('/Users/apple1/Desktop/编程/python/opencv/books/OpenCV 3计算机视觉 Python实现2/pycv-master/chapter5/cascades/haarcascade_frontalface_default.xml')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = face_classifier.detectMultiScale(gray, scaleFactor = 1.3, minSize = (40,40))
color = (255,0,0)

for (x,y,w,h) in faces:
	gray_face = gray[(y):(y+h), (x):(x+w)]
	print(np.shape(gray_face))
	gray_face = cv2.resize(gray_face, (48,144))
	gray_face = gray_face/255.0
	print(np.shape(gray_face))
	gray_face = np.expand_dims(gray_face, 0)
	print(np.shape(gray_face))
	gray_face = np.expand_dims(gray_face, -1)
	gray_face = np.reshape(gray_face, (1,48,48,3))
	print(np.shape(gray_face))
	emotion_label_arg = np.argmax(emotion_classifier.predict(gray_face))
	emotion = emotion_labels[emotion_label_arg]
	cv2.rectangle(img, (x+10, y+10), (x + h - 10, y + w - 10), (255, 255, 255), 2)
	img = cv2.putText(img, emotion, (x+w,y), cv2.FONT_HERSHEY_DUPLEX, 1.0, color, 3)


cv2.imshow('image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()




