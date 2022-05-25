import cv2
from threading import Thread, Event
import time
# import numpy as np


def write_frame(frame, text, data):
    return cv2.putText(frame, text, (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)


def process_frame(frame, type_process="normal"):
    if type_process == "normal":
        return frame
    elif type_process == "gray":
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    else:
        # convert frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        equ = cv2.equalizeHist(gray)

        # apply gaussian blur
        blur = cv2.GaussianBlur(equ, (5, 5), 0)

        # apply canny edge detection
        canny = cv2.Canny(blur, 50, 150)

    # return the result
        return canny


def detect(frame):
    return 0


def compare2cameras(cam1, cam2):
    while True:
        if cam1.event.is_set() and cam2.event.is_set():
            print("Both cameras are off! Exiting...")
            break
        if cam1.detect and cam2.detect:
            cam1.text = "Both cameras detected!"
            cam2.text = "Both cameras detected!"
            cam1.detect = False
            cam2.detect = False
            # usinf time waitn 0.1 seconds
            time.sleep(0.5)
            cam1.text = ""
            cam2.text = ""

        # else:
        #    print(f'Camera 1: {cam1.detect} Camera 2: {cam2.detect}')


class Camera(Thread):
    def __init__(self, cameraID, cam_name,
                 function_preprocess=process_frame,
                 function_detect=detect,
                 function_write=write_frame):
        super().__init__()
        self.cameraID = cameraID
        self.cam_name = cam_name
        self.function_preprocess = function_preprocess
        self.function_detect = function_detect
        self.function_write = function_write
        self.detect: bool = False
        self.event = Event()
        self.text = "Default"

    def run(self):
        run_camera(self, self.cameraID, self.cam_name)


def run_camera(camera_object, cameraID, cam_name) -> None:
    #  open camera using opencv
    cap = cv2.VideoCapture(cameraID)
    pila = []

    # loop camera until user presses 'q'
    while True:
        # read frame
        _, frame = cap.read()

        # display frame
        result = camera_object.function_preprocess(frame)
        # store 8 frames in a list
        pila.append(result)
        if len(pila) > 8:
            pila.pop(0)
        # sum of frames in pila
        sum_pila = sum(pila)
        # opening
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        sum_pila = cv2.morphologyEx(sum_pila, cv2.MORPH_CLOSE, kernel)
        # get info from the frame
        data = camera_object.function_detect(sum_pila)
        # draw things in the frame
        drawed = camera_object.function_write(sum_pila, camera_object.text, data)

        # show the frame
        cv2.imshow(cam_name, drawed)
        cv2.imshow('Frame', frame)

        # wait for user to press 'q'
        if cv2.waitKey(1) & 0xFF == ord('a'):
            camera_object.detect = True
            camera_object.text = "Camera detected!"

        if cv2.waitKey(1) & 0xFF == ord('q'):
            camera_object.event.set()
            break
