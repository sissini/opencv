import cv2

classifier = cv2.CascadeClassifier('/Users/apple1/Desktop/编程/python/opencv/books/OpenCV 3计算机视觉 Python实现2/pycv-master/chapter5/cascades/haarcascade_frontalface_default.xml')

img = cv2.imread('/Users/apple1/Desktop/编程/python/Tesseract_ORC/reisen.jpg')
imgCompose = cv2.imread('/Users/apple1/Desktop/编程/python/Tesseract_ORC/cmHat.jpeg')

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
color = (0, 255, 0)
faceRects = classifier.detectMultiScale(gray, scaleFactor= 1.2, minSize =(32,32))
if len(faceRects):
	for facerect in faceRects:
		x, y, w, h = facerect
		sp = imgCompose.shape
		imgComposeSizeH = int(sp[0]/sp[1]*w)
		if imgComposeSizeH > (y-20):
			imgComposeSizeH = y-20
		imgComposeSize = cv2.resize(imgCompose, (w, imgComposeSizeH), interpolation = cv2.INTER_NEAREST)
		top = y- imgComposeSizeH-20
		if top <= 0: top = 0
		rows, cols, channels = imgComposeSize.shape
		roi = img[top:top+rows, x:x+cols]

		# create a mask of logo
		img2gray = cv2.cvtColor(imgComposeSize, cv2.COLOR_RGB2GRAY)
		ret, mask = cv2.threshold(img2gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
		mask_inv = cv2.bitwise_not(mask)

		img_bg = cv2.bitwise_and(roi, roi, mask = mask)
		img_fg = cv2.bitwise_and(imgComposeSize, imgComposeSize, mask = mask_inv)

		dst = cv2.add(img_bg, img_fg)
		img[top:top+rows, x:x+cols] = dst

cv2.imshow('Image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

