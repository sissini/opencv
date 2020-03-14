import cv2 as cv

filename = 'jieke.jpg'
PATH = '/Users/travis/build/skvark/opencv-python/opencv/modules/objdetect/src/cascadedetect'

def detect(file):
	face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
	face_cascade.load(PATH+'/haarcascade_frontalface_default.xml')
	img = cv.imread(file)
	gray = cv.cvtColor(img, 0)
	faces = face_cascade.detectMultiScale(img, 1.3, 5)
	for (x,y,w,h) in faces:
		img = cv.rectangle(img, (x,y), (x+w, y+h), (255,0,0), 2)
	cv.namedWindow('fridends detect')
	cv.imshow('friends', img)
	cv.waitKey()

detect(filename)
