import face_recognition
from PIL import Image, ImageDraw

img = face_recognition.load_image_file('/Users/apple1/Desktop/编程/python/Tesseract_ORC/reisen.jpg')

face_landmarks_list = face_recognition.face_landmarks(img)

for flm in face_landmarks_list:
	facial_features = [ 'chin', 'left_eyebrow', 'right_eyebrow', 'nose_bridge', 'nose_tip',
        'left_eye', 'right_eye', 'top_lip', 'bottom_lip']
	pil_image = Image.fromarray(img)
	d = ImageDraw.Draw(pil_image)
	for ff in facial_features:
		d.line(flm[ff], fill(255, 0,0), width = 1)
	pil_image.show()

