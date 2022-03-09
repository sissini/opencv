import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt 



img = cv.imread('jjlin.jpg',0)
# IMREAD_GRAYCOLOR = 0
# IMREAD_COLOR =1
# IMREAD_UNCHANGED = -1

##显示图片##test
#cv.namedWindow('w',cv.WINDOW_NORMAL)	## 改变窗口大小
cv.imshow('w', img)
key = cv.waitKey(0)
if key == 27:	#esc exit
	cv.destroyAllWindows()
if key == ord('s'):
	cv.imwrite('jj_neu.jpg',img)
	cv.destroyAllWindows()
