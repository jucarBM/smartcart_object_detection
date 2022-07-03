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
# from pyzbar.pyzbar import decode
PATH_TO_MODEL_DIR = "models/fine_tuned_model_last"
# PATH_TO_MODEL_DIR = "../Barcode/models/fine_tuned_model_5000_ds"
PATH_TO_SAVE_MODEL = PATH_TO_MODEL_DIR + "/saved_model"
# PATH_TO_SAVE_MODEL = PATH_TO_MODEL_DIR
SHOW_VIDEO = True
TRESHOLD = 0.5
detect_fn = tf.saved_model.load(PATH_TO_SAVE_MODEL)

# Video Capture 
vid = cv2.VideoCapture(0)
# set resolution to 1920×1080, 3264 x 2448;
vid.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
vid.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
# Definimos ancho y alto
W = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
H = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
count = 0
class_name = ['none', 'leche_laive', 'chips_ahoy', 'jabon_bolivar']
while True:

    # Capture the video frame
    # by frame
    ret, frame = vid.read()

    # experimental
    # *************************************************************
    # final = frame
    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # equ = cv2.equalizeHist(gray)
    # threshold
    # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(5, 5))
    # final = clahe.apply(gray)   

    # *************************************************************

    final = frame

    if frame is None:
        break
    sku = "none"
    image_np = np.array(frame)

    input_tensor = tf.convert_to_tensor(image_np)
    input_tensor = input_tensor[tf.newaxis, ...]

    # Predecimos los objectos y clases de la imagen
    detections = detect_fn(input_tensor)
    detection_scores = np.array(detections["detection_scores"][0])
    # Realizamos una limpieza para solo obtener las clasificaciones mayores al umbral.
    detection_clean = [x for x in detection_scores if x >= TRESHOLD]
    # print(detections)

    # Recorremos las detecciones
    for x in range(len(detection_clean)):
        idx = int(detections['detection_classes'][0][x])
        name = class_name[idx]
        # Tomamos los bounding box
        ymin, xmin, ymax, xmax = np.array(
            detections['detection_boxes'][0][x])
        box = [xmin, ymin, xmax, ymax] * np.array([W, H, W, H])

        (startX, startY, endX, endY) = box.astype("int")
        '''cutImage = frame[startY:endY, startX:endX]
        barcodes_detected = decode(cutImage)
        for barcode in barcodes_detected:
            sku = str(barcode.data.decode("utf-8"))
            print(f'Detected {sku}')'''

        if SHOW_VIDEO:
            cv2.rectangle(frame, (startX, startY),
                          (endX, endY), (255, 0, 255), 2)
            cv2.putText(frame, str(round(100*detection_scores[x]))+" "+name, 
                        (startX, startY),
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        0.6, (0, 0, 255), 2)
        '''cv2.imwrite(
            f'Images/barcodes/{count}_{str(detection_scores[x])}_{sku}.png', cutImage)'''
        count += 1
    # Display the resulting frame
    if SHOW_VIDEO:
        resizedFrame = cv2.resize(frame,(0,0),fx = 0.90,fy = 0.90)
        cv2.imshow('frame', resizedFrame)
        # the 'q' button is set as the
        # quitting button you may use any
        # desired button of your choice
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# After the loop release the cap object
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()
