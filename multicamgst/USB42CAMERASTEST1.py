
#!/usr/bin/env python3
#
#  USB Camera - Simple



import sys

import cv2

window_title = "USB Camera"

# ASSIGN CAMERA ADRESS to DEVICE HERE!
pipeline = " ! ".join(["v4l2src device=/dev/video0",
                       "video/x-raw, width=640, height=480, framerate=30/1",
                       "videoconvert",
                       "video/x-raw, format=(string)BGR",
                       "appsink"
                       ])
               
pipeline1 = " ! ".join(["v4l2src device=/dev/video1",
                       "video/x-raw, width=640, height=480, framerate=30/1",
                       "videoconvert",
                       "video/x-raw, format=(string)BGR",
                       "appsink"
                       ])
                       
camset0 ='gst-launch-1.0 v4l2src device=/dev/video0 ! image/jpeg, width=640, height=480, framerate=15/1, format=MJPG ! nvv4l2decoder mjpeg=1 ! nvvidconv ! xvimagesink'          
camset1  = 'gst-launch-1.0 v4l2src device=/dev/video1 ! image/jpeg, width=640, height=480, framerate=15/1, format=MJPG ! nvv4l2decoder mjpeg=1 ! nvvidconv ! xvimagesink'          
             

video_capture = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
video_capture1 = cv2.VideoCapture(pipeline1, cv2.CAP_GSTREAMER)
    

while True:
	
	ret_val, frame = video_capture.read()
	ret_val1, frame1 = video_capture1.read()
	
	cv2.imshow('cam0', frame)
	cv2.imshow('cam1', frame1)
	
	cv2.moveWindow('cam0',0,0)
	cv2.moveWindow('cam0',0,500)                
				   
	if cv2.waitKey(1)==('q'):
		break

	 
video_capture.release()
video_capture1.release()
cv2.destroyAllWindows()
	
	
	          
'''
def show_camera(pipeline):
 # Full list of Video Capture APIs (video backends): https://docs.opencv.org/3.4/d4/d15/group__videoio__flags__base.html
    # For webcams, we use V4L2
    video_capture = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
    #video_capture1 = cv2.VideoCapture(pipeline1, cv2.CAP_GSTREAMER)

    if video_capture.isOpened() :
        try:
            window_handle = cv2.namedWindow(
                window_title, cv2.WINDOW_AUTOSIZE)
            # Window
            while True:
                ret_val, frame = video_capture.read()
                #ret_val1, frame1 = video_capture1.read()
                # Check to see if the user closed the window
                # Under GTK+ (Jetson Default), WND_PROP_VISIBLE does not work correctly. Under Qt it does
                # GTK - Substitute WND_PROP_AUTOSIZE to detect if window has been closed by user
                if cv2.getWindowProperty(window_title, cv2.WND_PROP_AUTOSIZE) >= 0:
                    cv2.imshow(window_title, frame)
                    
                else:
                    break
                keyCode = cv2.waitKey(10) & 0xFF
                # Stop the program on the ESC key or 'q'
                if keyCode == 27 or keyCode == ord('q'):
                    break

        finally:
            video_capture.release()
            cv2.destroyAllWindows()
    else:
        print("Error: Unable to open camera")

'''

'''
def show_camera(pipeline,pipeline1):
 # Full list of Video Capture APIs (video backends): https://docs.opencv.org/3.4/d4/d15/group__videoio__flags__base.html
    # For webcams, we use V4L2
    video_capture = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
    video_capture1 = cv2.VideoCapture(pipeline1, cv2.CAP_GSTREAMER)
    

	while True:
		
		ret_val, frame = video_capture.read()
		ret_val1, frame1 = video_capture1.read()
		
		cv2.imshow('cam0', frame)
		cv2.imshow('cam1', frame1)
		
		cv2.moveWindow('cam0',0,0)
		cv2.moveWindow('cam0',0,500)                
					   
		if keyCode == 27 or keyCode == ord('q'):
			break

	 
	video_capture.release()
	video_capture1.release()
	cv2.destroyAllWindows()

'''
