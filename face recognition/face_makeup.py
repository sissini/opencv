import face_recognition
from PIL import Image, ImageDraw

image = face_recognition.load_image_file('/Users/apple1/Desktop/编程/python/Tesseract_ORC/IMG_8917.jpeg')
# pil_image = Image.fromarray(image)
# pil_image.show('make up')

face_landmarks_list = face_recognition.face_landmarks(image)

for face_landmarks in face_landmarks_list:
	pil_image = Image.fromarray(image)
	d = ImageDraw.Draw(pil_image, 'RGBA')

	# eyebrow
	d.polygon(face_landmarks['left_eyebrow'], fill = (68, 54, 39 ,128))
	d.polygon(face_landmarks['right_eyebrow'], fill = (68, 54, 39 ,128))
	d.line(face_landmarks['left_eyebrow'], fill = (60,54,39,150), width=5)
	d.line(face_landmarks['right_eyebrow'], fill = (60,54,39,150), width=5)

	# lip
	d.polygon(face_landmarks['top_lip'], fill = (150, 0, 0 ,128))
	d.polygon(face_landmarks['bottom_lip'], fill = (150, 0, 0 ,128))
	d.line(face_landmarks['top_lip'], fill = (60,54,39,150), width=5)
	d.line(face_landmarks['bottom_lip'], fill = (60,54,39,150), width=5)

	# eye
	d.polygon(face_landmarks['left_eye'], fill=(255, 255, 255, 30))
	d.polygon(face_landmarks['right_eye'], fill=(255, 255, 255, 30))

	d.line(face_landmarks['left_eye'] + [face_landmarks['left_eye'][0]], fill = (0,0,0,100), width =6)
	d.line(face_landmarks['right_eye'], fill = (0,0,0,100), width =6)

	pil_image.show('make up')