import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt 



img = cv.imread('1.png',0)
img = img.resize(28,28)
size = img.size
print('size:', size)
# IMREAD_GRAYCOLOR = 0
# IMREAD_COLOR =1
# IMREAD_UNCHANGED = -1

##显示图片##
#cv.namedWindow('w',cv.WINDOW_NORMAL)	## 改变窗口大小
# cv.imshow('w', img)
# key = cv.waitKey(0)
# if key == 27:	#esc exit
# 	cv.destroyAllWindows()
# if key == ord('s'):
# 	cv.imwrite('jj_neu.jpg',img)
# 	cv.destroyAllWindows()



##plt显示图片
plt.imshow(img, cmap='gray', interpolation='bicubic')
# plt.xticks([])
# plt.yticks([])
# plt.plot([100,200,0],[500,100,100],'bo--', linewidth=5)
plt.show()

# ##制表
# x = np.linspace(-2,2,10)
# y = x**2
# plt.figure("test")
# plt.plot(x,y,'c',linewidth=2,label='square function')
# #限制x y轴的显示范围
# # plt.xlim(-1,2)
# # plt.ylim(-1,3)
# plt.xlabel('x axis')
# plt.ylabel('y axis')
# plt.xticks(np.linspace(-2,2,5))
# plt.yticks([0,1,2,4],['mini','low middle','high middle','max'])
# plt.legend()
# #labels set
# ax = plt.gca()	##get curent axis
# for label in ax.get_xticklabels() + ax.get_yticklabels():
# 	label.set_fontsize(10)
# 	label.set_bbox(dict(facecolor='blue',edgecolor='None', alpha=0.3))

# plt.show()
# #################

# ## line，rec，圆，椭圆，多边形，文字
# img = np.zeros((512,512,3),np.uint8)
# pts=np.array([[10,5],[20,30],[70,20],[50,10]],np.int32)
# pts = pts.reshape((-1,1,2))
# font = cv.FONT_HERSHEY_SIMPLEX

# cv.line(img,(0,0),(100,100),(255,0,0),10)
# cv.rectangle(img,(0,0),(100,100),(255,0,0),10)
# cv.circle(img,(50,50),20,(0,0,255),-1)
# cv.ellipse(img,(200,250),(50,20),0,0,180,(100,0,50),-1)
# cv.polylines(img,[pts],False,(0,255,255)) ## False 不闭合 true 曲线闭合
# cv.putText(img,'Test',(400,400),font,3,(0,255,0),10,cv.LINE_AA) # 3--字体大小  10--字体粗细

# cv.namedWindow('image',cv.WINDOW_NORMAL)
# cv.resizeWindow('image',1000,1000)
# cv.imshow('image',img)
# cv.waitKey(0)
# CV.destroyAllWindows()



