import cv2
import dlib

detector = dlib.get_frontal_face_detector()


def discern(img):
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	dets = detector(gray, 1)
	for face in dets:
		left = face.left()
		top = face.top()
		right = face.right()
		bottom = face.bottom()
		cv2.rectangle(img, (left, top), (right, bottom), (0, 0, 255), 2)
		cv2.imshow('Image', img)

cap = cv2.VideoCapture(0)
while (1):
	rec, img = cap.read()
	discern(img)
	if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()

 