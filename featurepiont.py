import cv2 as cv

img = cv.imread('badenbaden.png')
gray= cv.cvtColor(img, 0)

##sift
sift = cv.xfeatures2d.SIFT_create(8000)
keypoints, descriptor = sift.detectAndCompute(gray, None)

img = cv.drawKeypoints(image=img, outImage=img, keypoints=keypoints, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS, color=(255,0,0))
cv.imshow('sift', img)



##surf
surf = cv.xfeatures2d.SURF_create(8000)
keypoints1, descriptor1 = sift.detectAndCompute(gray, None)
img = cv.drawKeypoints(image=img, outImage=img, keypoints=keypoints1, flags=4, color=(255,0,0))
cv.imshow('surf', img)
cv.waitKey()
cv.destroyALLWindows()