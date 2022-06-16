from itertools import count
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from imutils.video import VideoStream
from imutils.video import FPS
from utils.centroidtracker import CentroidTracker
from utils.trackableobject import TrackableObject
import tensorflow as tf
from pyzbar.pyzbar import decode
PATH_TO_MODEL_DIR = "models/converted"
# PATH_TO_MODEL_DIR = "../Barcode/models/fine_tuned_model_5000_ds"
# PATH_TO_SAVE_MODEL = PATH_TO_MODEL_DIR + "/saved_model"
PATH_TO_SAVE_MODEL = PATH_TO_MODEL_DIR
SHOW_VIDEO = True
TRESHOLD = 0.7
detect_fn = tf.saved_model.load(PATH_TO_SAVE_MODEL)

# Video Capture 
vid = cv2.VideoCapture('v4l2src device=/dev/video0 io-mode=2 ! image/jpeg, width=1920, height=1080, framerate=30/1 !  nvjpegdec ! video/x-raw ! videoconvert ! video/x-raw,format=BGR ! appsink', cv2.CAP_GSTREAMER)
vid1 = cv2.VideoCapture('v4l2src device=/dev/video1 io-mode=2 ! image/jpeg, width=1920, height=1080, framerate=30/1 !  nvjpegdec ! video/x-raw ! videoconvert ! video/x-raw,format=BGR ! appsink', cv2.CAP_GSTREAMER)


# set resolution to 1920×1080, 3264 x 2448;
# vid.set(cv2.CAP_PROP_FRAME_WIDTH, 3264)
# vid.set(cv2.CAP_PROP_FRAME_HEIGHT, 2448)
# Definimos ancho y alto
W = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
H = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))

#------------------------------------
W1 = int(vid1.get(cv2.CAP_PROP_FRAME_WIDTH))
H1 = int(vid1.get(cv2.CAP_PROP_FRAME_HEIGHT))

#------------------------
count = 0
while True:

    # Capture the video frame
    # by frame
    ret, frame = vid.read();ret1, frame1 = vid1.read()	
	
    # experimental
    # *************************************************************
    # final = frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #--------------------------------------
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    
    #------------------------------------
    # equ = cv2.equalizeHist(gray)
    # threshold
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(5, 5))
    final = clahe.apply(gray)
    #-final1
    final1 = clahe.apply(gray1) 

    # *************************************************************

    if frame is None and frame1 is None:
        break
    sku = "none"
    #---------------iamge and image1
    image_np = np.array(frame);	image_np1 = np.array(frame1)
	
	
    input_tensor = tf.convert_to_tensor(image_np)
    input_tensor1 = tf.convert_to_tensor(image_np1)
    
    
    input_tensor = input_tensor[tf.newaxis, ...]
    input_tensor1 = input_tensor1[tf.newaxis, ...]

    # Predecimos los objectos y clases de la imagen
    detections = detect_fn(input_tensor)
    detections1 = detect_fn(input_tensor1)
    
    
    detection_scores = np.array(detections["detection_scores"][0])
    detection_scores1 = np.array(detections1["detection_scores"][0])
    
    
    # Realizamos una limpieza para solo obtener las clasificaciones mayores al umbral.
    detection_clean = [x for x in detection_scores if x >= TRESHOLD]
    detection_clean1 = [x for x in detection_scores1 if x >= TRESHOLD]
    
    
    # Recorremos las detecciones
    for x in range(len(detection_clean)):
        idx = int(detections['detection_classes'][0][x])
        # Tomamos los bounding box
        ymin, xmin, ymax, xmax = np.array(
            detections['detection_boxes'][0][x])
        box = [xmin, ymin, xmax, ymax] * np.array([W, H, W, H])

        (startX, startY, endX, endY) = box.astype("int")
        cutImage = frame[startY:endY, startX:endX]
        barcodes_detected = decode(cutImage)
        for barcode in barcodes_detected:
            sku = str(barcode.data.decode("utf-8"))
            print(f'Detected {sku}')

        if SHOW_VIDEO:
            cv2.rectangle(frame, (startX, startY),
                          (endX, endY), (255, 0, 255), 2)
            cv2.putText(frame, str(round(100*detection_scores[x]))+" "+str(sku), 
                        (startX, startY),
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        0.6, (0, 0, 255), 2)
        cv2.imwrite(
            f'Images/barcodes/{count}_{str(detection_scores[x])}_{sku}.png', cutImage)
        count += 1
        
   for x in range(len(detection_clean1)):
        idx1 = int(detections1['detection_classes'][0][x])
        # Tomamos los bounding box
        ymin1, xmin1, ymax1, xmax1 = np.array(
            detections1['detection_boxes'][0][x])
        box1 = [xmin1, ymin1, xmax1, ymax1] * np.array([W1, H1, W1, H1])

        (startX1, startY1, endX1, endY1) = box1.astype("int")
        cutImage1 = frame[startY1:endY1, startX1:endX1]
        barcodes_detected1 = decode(cutImage1)
        for barcode1 in barcodes_detected1:
            sku1 = str(barcode1.data.decode("utf-8"))
            print(f'Detected {sku1}')

        if SHOW_VIDEO:
            cv2.rectangle(frame1, (startX1, startY1),
                          (endX1, endY1), (255, 0, 255), 2)
            cv2.putText(frame1, str(round(100*detection_scores1[x]))+" "+str(sku1), 
                        (startX1, startY1),
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        0.6, (0, 0, 255), 2)
        cv2.imwrite(
            f'Images/barcodes/{count}_{str(detection_scores1[x])}_{sku1}.png', cutImage1)
        count += 1
       
    # Display the resulting frame
    if SHOW_VIDEO:
        resizedFrame = cv2.resize(frame,(0,0),fx = 0.20,fy = 0.20)
        resizedFrame1 = cv2.resize(frame1,(0,0),fx = 0.20,fy = 0.20)
        
        
        cv2.imshow('frame', resizedFrame)
        cv2.imshow('frame1', resizedFrame1)
        cv2.moveWindow('frame',0,0)
		cv2.moveWindow('frame1',0,500)  
        
        
        # the 'q' button is set as the
        # quitting button you may use any
        # desired button of your choice
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# After the loop release the cap object
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()
