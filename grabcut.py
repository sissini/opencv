import numpy as np 
import cv2
from matplotlib import pyplot as plt 

img = cv2.imread('jjlin.jpg')
mask = cv2.zeros(img.shape(None,2), np.uint8)

bgmodel = cv2.zeros((1,65), np.float64)
fgmodel = cv2.zeros((1,65), np.float64)

rect = (100,50,400, 200)
cv2.grabCut(img, mask, rect, bgmodel, fgmodel, 5, cv2.GC_INIT_WITH_RECT)

mask2 = np.where((mask==2)|(mask==0), 0,1).astype('uint8')
img = img*mask2[:,:,np.newaxis]

plt.subplot(121)