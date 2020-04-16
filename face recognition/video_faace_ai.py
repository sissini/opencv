import cv2

def discern(img):
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	cap = cv2.CascadeClassifier('/Users/apple1/PycharmProjects/MY_FIRST/venv/lib/python3.7/site-packages/OpenCV-android-sdk/sdk/etc/haarcascades/haarcascade_frontalface_default.xml')
	facerects = cap.detectMultiScale(gray, scaleFactor = 1.2, minSize = (50,50))
	if len(facerects):
		for facerect in facerects:
			x,y,w,h = facerect
			cv2.rectangle(img, (x,y), (x+w, y+h), (0,0,255),2)
	cv2.imshow('Image', img)


cap = cv2.VideoCapture(0)

while (1):
	ret, img = cap.read()
	discern(img)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break
cap.release()
cv2.destroyAllWindows()



