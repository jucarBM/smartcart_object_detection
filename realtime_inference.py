from itertools import count
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from imutils.video import VideoStream
from imutils.video import FPS
from centroidtracker import CentroidTracker
from trackableobject import TrackableObject
import tensorflow as tf

PATH_TO_MODEL_DIR = "models/fine_tuned_model_5000_ds"
PATH_TO_SAVE_MODEL = PATH_TO_MODEL_DIR + "/saved_model"
SHOW_VIDEO = True

TRESHOLD = 0.25
detect_fn = tf.saved_model.load(PATH_TO_SAVE_MODEL)

vid = cv2.VideoCapture(0)
# Definimos ancho y alto
W = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
H = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
count = 0
while(True):

    # Capture the video frame
    # by frame
    ret, frame = vid.read()

    if frame is None:
        break

    image_np = np.array(frame)

    input_tensor = tf.convert_to_tensor(image_np)
    input_tensor = input_tensor[tf.newaxis, ...]

    # Predecimos los objectos y clases de la imagen
    detections = detect_fn(input_tensor)
    detection_scores = np.array(detections["detection_scores"][0])
    # Realizamos una limpieza para solo obtener las clasificaciones mayores al umbral.
    detection_clean = [x for x in detection_scores if x >= TRESHOLD]
    # Recorremos las detecciones
    for x in range(len(detection_clean)):
        idx = int(detections['detection_classes'][0][x])
        # Tomamos los bounding box
        ymin, xmin, ymax, xmax = np.array(
            detections['detection_boxes'][0][x])
        box = [xmin, ymin, xmax, ymax] * np.array([W, H, W, H])

        (startX, startY, endX, endY) = box.astype("int")
        if SHOW_VIDEO:
            cv2.rectangle(frame, (startX, startY),
                          (endX, endY), (255, 0, 255), 2)
            cv2.putText(frame, str(detection_scores[x])+" "+str(idx), (startX, startY),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            cutImage = frame[startY:endY, startX:endX]
        cv2.imwrite(
            f'Images/barcodes/{count}_{str(detection_scores[x])}_{idx}.jpeg', cutImage)
        count += 1
    # Display the resulting frame
    if SHOW_VIDEO:
        cv2.imshow('frame', frame)
        # the 'q' button is set as the
        # quitting button you may use any
        # desired button of your choice
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# After the loop release the cap object
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()
