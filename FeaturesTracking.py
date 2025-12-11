# press 'Setup Parameters' in the OP to call this function to re-create the parameters.
def onSetupParameters(scriptOp):
	# create a custom page
	page = scriptOp.appendCustomPage('Good Features')

	# create a custom TOP reference parameter
	topPar = page.appendTOP('Top', label='TOP (monochrome)')

	# create a custom parameter to specify number of features to detect
	p = page.appendInt('Features', label='Number of Features')
	p[0].default = 25
	p[0].normMin = 1
	p[0].normMax = 250

	# create a custom parameter to specify minimum quality level
	# under which detected features would be rejected
	p = page.appendFloat('Quality', label='Minimum Quality Level')
	p[0].default = 0.01
	p[0].normMin = 0.001
	p[0].normMax = 1

	# create a custom parameter to specify the minimum distance
	# between detected features
	p = page.appendInt('Distance', label='Minimum Distance')
	p[0].default = 10
	p[0].normMin = 1
	p[0].normMax = 1200
	return

# called whenever custom pulse parameter is pushed
def onPulse(par):
	print('Find Features')
	return
	
import numpy as np
import cv2

def onCook(scriptOp):
	scriptOp.clear()
	
	# read in parameters to see how many features to detect
	topRef = scriptOp.par.Top.eval()
	maxCorners = int(scriptOp.par.Features)
	qualityLevel = float(scriptOp.par.Quality)
	minDistance = float(scriptOp.par.Distance)
	
	# default values
	xVals = []
	yVals = []
	corners = []
	
	if topRef:
		# read top as numpyArray
		img = topRef.numpyArray()
		
		# since we are reading from a gray scale TOP, throw out everything but red channel
		# we also can skip the cv2.cvtColor function you would see here otherwise for converting a color image to gray scale
		img = img[:,:,:1]
		
		# run goodFeaturesToTrack openCV function 
		# before
		# https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_feature2d/py_shi_tomasi/py_shi_tomasi.html
		# after
		# https://docs.opencv.org/3.4/dd/d1a/group__imgproc__feature.html#ga1d6bb77486c8f92d79c8793ad995d541
		# corners = cv2.goodFeaturesToTrack(img, int(features), int(1.0*quality), int(distance))
		# After (fixed):
		qualityLevel = max(0.01, qualityLevel) if qualityLevel <= 0 else qualityLevel
		minDistance = max(0, minDistance)
		maxCorners = max(0, maxCorners)
		corners = cv2.goodFeaturesToTrack(img, maxCorners, qualityLevel, minDistance)
		
		# slice array to have x and y positions split into 2 variables
		xVals = corners[:,:,0:1]
		yVals = corners[:,:,1:2]
	
	# setup the scriptOp with 2 channels
	# also set length to number of features that were detected
	scriptOp.rate = me.time.rate
	scriptOp.numSamples = len(corners)
	tx = scriptOp.appendChan('tx')
	ty = scriptOp.appendChan('ty')
	
	# assign values to channels
	tx.vals = xVals
	ty.vals = yVals
	return