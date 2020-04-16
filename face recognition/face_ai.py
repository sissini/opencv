import cv2 

filepath = '/Users/apple1/Desktop/编程/python/Tesseract_ORC/reisen.jpg'
img = cv2.imread(filepath)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# opencv face recognition classifier
classifier = cv2.CascadeClassifier('/Users/apple1/PycharmProjects/MY_FIRST/venv/lib/python3.7/site-packages/OpenCV-android-sdk/sdk/etc/haarcascades/haarcascade_frontalface_default.xml')
color = (0,0,255)
facerects = classifier.detectMultiScale(gray, scaleFactor =1.2, minSize = (32,32))
if len(facerects):
	for facerect in facerects:
		x,y,w,h =facerect
		cv2.rectangle(img, (x,y), (x+w, y+h), color, 2)


cv2.imshow('Image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

