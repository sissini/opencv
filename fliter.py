import cv2 as cv
import numpy as np 
import matplotlib.pyplot as plt 
from scipy import ndimage as nd

# class VConvolutionfliter(object):
# 	"""a filter that applies a convolution to v """
# 	def __init__(self, kernel):
# 		self.kernel = kernel
# 	def apply(self, src, dst):
# 		"""apply the filter with a bgr/gray source/destination"""
# 		cv.filter2D(src, -1, self_kernel, dst)

# class SharpenFilter(VConvolutionfliter):
# 	"""docstring for SharpenFilter"""
# 	def __init__(self):
# 		kernel = np.array([[-1,-1,-1],[-1,9,-1],[-1,-1,-1]])
# 		VConvolutionfliter.__init__(self, kernel)

# class FindEdgesFilter(VConvolutionfliter):
# 	"""docstring for FindEdgesFilter"""
# 	def __init__(self):
# 		kernel = np.array([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]])
# 		VConvolutionfliter.__init__(self, kernel)
		




 
# # kernel_3x3 = np.array([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]])
# # k3 = nd.convolve(img, kernel_3x3)
# # blurred = cv.GaussianBlur(img,(11,11),0)
# # g_hpf = img - blurred
# # median = cv.medianBlur(img,5)
# # blur = cv.bilateralFilter(img, 9,75,75)
# # cv.imshow('median image', median)
# # cv.imshow('blur image', blur)
# # cv.imshow('hpf image', g_hpf)
# # cv.imshow('blurred image', blurred)
# # cv.imshow('convolve', k3)
# # cv.imshow('laplace', lap)

img = cv.imread('./images/jjlin.jpg',0) 
img_canny = cv.Canny(img, 200, 300)
# graysrc = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# lap = cv.Laplacian(graysrc, cv.CV_8U)
# normalizedInverseAlpha = (1.0/255)*(255-graysrc)

# channels = cv.split(img)
# for channel in channels:
# 	channel[:] = channel*normalizedInverseAlpha
# neu_img = cv.merge(channels)

# cv.imshow('neu image', neu_img)
cv.imshow('image Canny', img_canny)
cv.waitKey()
cv.destroyAllWindows()