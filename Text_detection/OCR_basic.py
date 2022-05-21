import cv2
import numpy as np
import pytesseract
import keras_ocr

# show loop frames from webcam using opencv

vs = cv2.VideoCapture(0)
count = 0
while True:
    # Leemos el primer frame
    ret, frame = vs.read()

    # Si ya no hay más frame
    if frame is None:
        break
    # cv2.imwrite(
    #    f'Images/analysis/{count}.png', frame)
    # count += 1
    # *********************************
    # Processing of frames goes here
    
    # *********************************

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# After the loop release the cap object
vs.release()
# Destroy all the windows
cv2.destroyAllWindows()