import cv2
import numpy as np

import enviouart.py as ipo

#!/usr/bin/python3
import time
import serial


# Video Capture 
capture = cv2.VideoCapture('v4l2src device=/dev/video0 io-mode=2 ! image/jpeg, width=1920, height=1080, framerate=30/1 !  nvjpegdec ! video/x-raw ! videoconvert ! video/x-raw,format=BGR ! appsink', cv2.CAP_GSTREAMER)
capture1 = cv2.VideoCapture('v4l2src device=/dev/video1 io-mode=2 ! image/jpeg, width=1920, height=1080, framerate=30/1 !  nvjpegdec ! video/x-raw ! videoconvert ! video/x-raw,format=BGR ! appsink', cv2.CAP_GSTREAMER)


# History, Threshold, DetectShadows 
# fgbg = cv2.createBackgroundSubtractorMOG2(150, 1000, True)
fgbg = cv2.createBackgroundSubtractorMOG2(200, 200, True)

# Keeps track of what frame we're on
frameCount = 0
frameCount1 = 0


serial_port = serial.Serial(
    port="/dev/ttyTHS1",
    baudrate=115200,
    bytesize=serial.EIGHTBITS,
    parity=serial.PARITY_NONE,
    stopbits=serial.STOPBITS_ONE,
)
# Wait a second to let the port initialize
time.sleep(1)



def mensajitoxd(sku):
	serial_port.write("Hola".encode())
	print("te envie hola")
	data  = serial_port.readline(5)/dev/ttyTHS1
	data  =data.decode()
	datastr  =  str(data)
	print(datastr)
	if datastr  =="Hola2":
		
		serial_port.write(sku.encode())
		print("entro")
		
		
while(1):
	# Return Value and the current frame
	ret, frame = capture.read()
	ret, frame1 = capture1.read()

	#  Check if a current frame actually exist
	if not ret:
		break

	frameCount += 1
	frameCount1 += 1
	# Resize the frame
	resizedFrame = cv2.resize(frame, (0, 0), fx=0.2, fy=0.2)
	resizedFrame1 = cv2.resize(frame1, (0, 0), fx=0.2, fy=0.2)
	# Get the foreground mask
	fgmask = fgbg.apply(resizedFrame)
	fgmask1 = fgbg.apply(resizedFrame1)
    # ------------------------------------------------
	gauss_filter = cv2.GaussianBlur(fgmask,(5,5),3)
	median_blur = cv2.medianBlur(fgmask,5)
    
	kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(15,15))
    # cierre = cv2.morphologyEx(gauss_filter,cv2.MORPH_CLOSE,kernel) 
	# dilate = cv2.dilate(gauss_filter,None,iterations=2) 
    # # ----------------------------------------------------------------
	# Count all the non zero pixels within the mask
	count = np.count_nonzero(fgmask)
	count1 = np.count_nonzero(fgmask1)
	print('Frame: %d, Pixel Count: %d' % (frameCount, count))

	# Determine how many pixels do you want to detect to be considered "movement"
	
	if ((frameCount > 1 and count > 3000 and count < 4000) or ( frameCount1 > 1  and count1 > 3000 and count1 < 4000)):
        # find contours or continuous white blobs in the image
		contours, hierarchy = cv2.findContours(fgmask.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
		contours1, hierarchy1 = cv2.findContours(fgmask1.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

		# find the index of the largest contour
		if len(contours) > 0:
			areas = [cv2.contourArea(c) for c in contours]
			max_index = np.argmax(areas)
			cnt=contours[max_index]   

			# draw a bounding box/rectangle around the largest contour
			x,y,w,h = cv2.boundingRect(cnt)
			cv2.rectangle(resizedFrame,(x,y),(x+w,y+h),(0,255,0),2)
			area = cv2.contourArea(cnt)

			# print area to the terminal
			print(area)
		
			# add text to the frame
			cv2.putText(resizedFrame, "Largest Contour", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
			mensajitoxd(detect)
			
			
		if len(contours1) > 0:
			areas1 = [cv2.contourArea(c) for c in contours1]
			max_index1 = np.argmax(areas1)
			cnt1=contours1[max_index1]   

			# draw a bounding box/rectangle around the largest contour
			x1,y1,w1,h1 = cv2.boundingRect(cnt1)
			cv2.rectangle(resizedFrame1,(x1,y1),(x1+w1,y1+h1),(0,255,0),2)
			area1 = cv2.contourArea(cnt1)

			# print area to the terminal
			print(area1)
		
			# add text to the frame
			cv2.putText(resizedFrame1, "Largest Contour", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
			mensajitoxd(detect)
		print('detect')
		cv2.putText(resizedFrame, 'detect', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
		cv2.putText(resizedFrame1, 'detect', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

	cv2.imshow('Frame', resizedFrame)
	cv2.imshow('Frame1', resizedFrame1)
	cv2.imshow('Mask', fgmask)
	cv2.imshow('Mask1', fgmask1)

	cv2.moveWindow('Frame',0,0)
	cv2.moveWindow('Frame1',0,500)
	cv2.moveWindow('Mask',500,0)
	cv2.moveWindow('Mask1',500,500)

	k = cv2.waitKey(1) & 0xff
	if k == 27:
		break

capture.release()
capture1.release()
cv2.destroyAllWindows()
