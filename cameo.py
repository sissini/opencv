import cv2 as cv
import filters as fl 
from managers import WindowManager, CaptureManager 
import depth

class Cameo(object):
	""tring for Cameo"""
	def __init__(self):
		self._windowsManager = WindowManager('cameo', self.onKeypress)
		self._captureManager = CaptureManager(cv.VideoCapture(0), self._windowsManager, True)
		self._curveFilter = filters.BGRPortraCurveFilter()

	def run(self):
		self._windowsManager.createWindow()
		while self._windowsManager.isWindowCreated:
			self._captureManager.enterFrame()
			frame = self._captureManager.frame

			filters.strokrEdges(frame, frame)
			self._curveFilter.apply(frame, frame)
			self._captureManager.exitFrame()
			self._windowsManager.processEvents()