import cv2
import numpy as np

# Video Capture 
capture = cv2.VideoCapture(1)


# History, Threshold, DetectShadows 
# fgbg = cv2.createBackgroundSubtractorMOG2(150, 1000, True)
fgbg = cv2.createBackgroundSubtractorMOG2(200, 200, True)

# Keeps track of what frame we're on
frameCount = 0

while(1):
	# Return Value and the current frame
	ret, frame = capture.read()

	#  Check if a current frame actually exist
	if not ret:
		break

	frameCount += 1
	# Resize the frame
	resizedFrame = cv2.resize(frame, (0, 0), fx=0.50, fy=0.50)

	# Get the foreground mask
	fgmask = fgbg.apply(resizedFrame)
    # ------------------------------------------------
	gauss_filter = cv2.GaussianBlur(fgmask,(5,5),3)
	median_blur = cv2.medianBlur(fgmask,5)

	kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(15,15))
	cierre = cv2.morphologyEx(gauss_filter,cv2.MORPH_CLOSE,kernel) 
    # ----------------------------------------------------------------
	# Count all the non zero pixels within the mask
	count = np.count_nonzero(cierre)

	print('Frame: %d, Pixel Count: %d' % (frameCount, count))

	# Determine how many pixels do you want to detect to be considered "movement"
	
	if (frameCount > 1 and count > 5000 and count < 9000):
		print('detect')
		cv2.putText(resizedFrame, 'detect', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

	cv2.imshow('Frame', resizedFrame)
	cv2.imshow('Mask', cierre)


	k = cv2.waitKey(1) & 0xff
	if k == 27:
		break

capture.release()
cv2.destroyAllWindows()
