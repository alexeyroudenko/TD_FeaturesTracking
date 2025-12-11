# press 'Setup Parameters' in the OP to call this function to re-create the parameters.
def onSetupParameters(scriptOp):
	# create a custom page
	page = scriptOp.appendCustomPage('Good Features')

	# create a custom TOP reference parameter
	topPar = page.appendTOP('Top', label='TOP (monochrome)')
	topPar.default = "img"
	
	topPrevPar = page.appendTOP('Topprevious', label='TOP Previous	 (monochrome)')
	topPrevPar.default = "cache"

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

	p = page.appendToggle('Useharris', label='Use Harris Corner Detection')
	p[0].default = False

	# p = page.appendToggle('ShowFeatures', label='Show Features')
	# p[0].default = False

	p1 = page.appendToggle('Findfeatures', label='Find Features')
	p1[0].default = True

	return

# called whenever custom pulse parameter is pushed
def onPulse(par):
	print('Find Features')
	return
	
import numpy as np
import cv2

def onCook(scriptOp):
	# read in parameters to see how many features to detect
	topRef = scriptOp.par.Top.eval()
	topPrevRef = scriptOp.par.Topprevious.eval()
	maxCorners = int(scriptOp.par.Features)
	qualityLevel = float(scriptOp.par.Quality)
	minDistance = float(scriptOp.par.Distance)
	useHarris = bool(scriptOp.par.Useharris)
	# showFeatures = bool(scriptOp.par.ShowFeatures)
	findFeatures = bool(scriptOp.par.Findfeatures)

	# Get previous points from channels BEFORE clearing (for optical flow tracking)
	prevPts = None
	if not findFeatures:
		try:
			# Try different methods to access channels
			tx = None
			ty = None
			
			# Method 1: Direct access
			try:
				tx = scriptOp['tx']
				ty = scriptOp['ty']
			except:
				pass
			
			# Method 2: Via chans dictionary
			if tx is None or ty is None:
				try:
					if hasattr(scriptOp, 'chans') and 'tx' in scriptOp.chans and 'ty' in scriptOp.chans:
						tx = scriptOp.chans['tx']
						ty = scriptOp.chans['ty']
				except:
					pass
			
			# Method 3: Via channel index
			if tx is None or ty is None:
				try:
					chans = list(scriptOp.chans)
					if len(chans) >= 2:
						tx = chans[0]
						ty = chans[1]
				except:
					pass
			
			if tx is not None and ty is not None:
				txVals = tx.vals if hasattr(tx, 'vals') else []
				tyVals = ty.vals if hasattr(ty, 'vals') else []
				
				if len(txVals) > 0 and len(tyVals) > 0:
					# Convert to format needed for calcOpticalFlowPyrLK: (N, 1, 2)
					prevPts = np.zeros((len(txVals), 1, 2), dtype=np.float32)
					prevPts[:, 0, 0] = np.array(txVals).flatten()
					prevPts[:, 0, 1] = np.array(tyVals).flatten()
					# print(f"Found {len(prevPts)} previous points for tracking")
		except Exception as e:
			prevPts = None
			print(f"Error getting previous points: {e}")
	
	scriptOp.clear()
	
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
		# Convert to 2D grayscale and ensure uint8 format
		img = np.squeeze(img)
		if img.dtype != np.uint8:
			# Normalize if float (0.0-1.0) or scale if other format
			if img.dtype == np.float32 or img.dtype == np.float64:
				img = (img * 255).astype(np.uint8)
			else:
				img = img.astype(np.uint8)
		
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
		
		if findFeatures:
			# Detect new features
			corners = cv2.goodFeaturesToTrack(img, maxCorners, qualityLevel, minDistance, useHarrisDetector=useHarris)
		else:
			# Track existing features using Lucas-Kanade optical flow
			if topPrevRef and prevPts is not None and len(prevPts) > 0:
				# Read previous frame
				imgPrev = topPrevRef.numpyArray()
				imgPrev = imgPrev[:,:,:1]
				# Convert to 2D grayscale and ensure uint8 format
				imgPrev = np.squeeze(imgPrev)
				if imgPrev.dtype != np.uint8:
					# Normalize if float (0.0-1.0) or scale if other format
					if imgPrev.dtype == np.float32 or imgPrev.dtype == np.float64:
						imgPrev = (imgPrev * 255).astype(np.uint8)
					else:
						imgPrev = imgPrev.astype(np.uint8)
				
				# Calculate optical flow using Lucas-Kanade method
				nextPts, status, err = cv2.calcOpticalFlowPyrLK(
					imgPrev, img, prevPts, None,
					winSize=(15, 15),
					maxLevel=2,
					criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
				)
				
				# Filter out points that failed to track (status == 0)
				good = status.ravel() == 1
				if np.any(good):
					corners = nextPts[good]
				else:
					corners = None
			else:
				corners = None
		
		# slice array to have x and y positions split into 2 variables
		if corners is not None and len(corners) > 0:
			xVals = corners[:,:,0:1]
			yVals = corners[:,:,1:2]
		else:
			xVals = np.array([])
			yVals = np.array([])
			corners = np.array([])
	
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