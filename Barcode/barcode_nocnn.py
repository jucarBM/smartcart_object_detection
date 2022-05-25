from itertools import count
import numpy as np
import cv2
import tensorflow as tf
from pyzbar.pyzbar import decode
# PATH_TO_MODEL_DIR = "models/fine_tuned_model_5000_ds"
PATH_TO_MODEL_DIR = "../models/fine_tuned_model_5000_ds"
PATH_TO_SAVE_MODEL = PATH_TO_MODEL_DIR + "/saved_model"
SHOW_VIDEO = True

TRESHOLD = 0.7
detect_fn = tf.saved_model.load(PATH_TO_SAVE_MODEL)

vid = cv2.VideoCapture(0)
 # set resolution to 1920×1080, 3264 x 2448;
vid.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
vid.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
# Definimos ancho y alto
W = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
H = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
count = 0
while(True):

    # Capture the video frame
    # by frame
    ret, frame = vid.read()

    # experimental
    # *************************************************************
    # final = frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # final = cv2.equalizeHist(gray)
    # threshold
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(5, 5))
    final = clahe.apply(gray)   

    # *************************************************************

    if frame is None:
        break
    sku = "none"
    image_np = np.array(final)

    barcodes_detected = decode(image_np)
    for barcode in barcodes_detected:
        sku = str(barcode.data.decode("utf-8"))
        print(f'Detected {sku}')
        (x, y, w, h) = barcode.rect
        cutImage = final[y:y+h, x:x+w]
        if SHOW_VIDEO:
            cv2.rectangle(frame, (x, y-10), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(frame, str(sku), 
                        (x, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        0.6, (0, 0, 255), 2)
        cv2.imwrite(
            f'../Images/barcodes/{count}_{sku}.png', cutImage)
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
