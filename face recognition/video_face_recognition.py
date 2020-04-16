import cv2 
import face_recognition
import os

path = '/Users/apple1/Desktop/编程/python/opencv/faceRecognition/images'
total_image_name = []
total_face_encoding = []

for filename in os.listdir(path):
	print(path + '/' +filename)
	image = face_recognition.load_image_file(path+'/'+filename)
	total_face_encoding.append(face_recognition.face_encodings(image)[0])
	filename = filename[:len(filename)-5]
	print(filename)
	total_image_name.append(filename)

cap = cv2.VideoCapture(0)
while(1):
	ret,frame = cap.read()
	face_locations = face_recognition.face_locations(frame)
	face_encodings = face_recognition.face_encodings(frame, face_location)

	for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
		for i,v in enumerate(total_face_encoding):
			match = face_recognition.compare_faces([v], face_encoding, tolerance = 0.5)
			name = 'unknown'
			if match[0]:
				name = total_image_name[i]
				break
		cv2.rectangle(frame, (left, top), (right, bottom), (0,0,255), cv2.FILLED)
		font = cv2.FONT_HERSHEY_DUPLEX
		cv2.putText(frame, name, (left+6, bottom -6), font, 1.0, (255,0,0), 1)

	cv2.imshow('video', frame)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break


cap.release()
cv2.destroyAllWindows()




