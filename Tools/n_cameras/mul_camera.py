import cv2
from threading import Thread, Event
import time
# import numpy as np


class Camera(Thread):
    def __init__(self, cameraID, cam_name, type_process="normal"):
        super().__init__()
        self.cameraID = cameraID
        self.cam_name = cam_name
        self.type_process = type_process
        self.detect: bool = False
        self.event = Event()
        self.text = "Default"

    def run(self):
        run_camera(self, self.cameraID, self.cam_name, self.type_process)


def run_camera(camera_object, cameraID, cam_name, type_process) -> None:
    #  open camera using opencv
    cap = cv2.VideoCapture(cameraID)

    # loop camera until user presses 'q'
    while True:
        # read frame
        _, frame = cap.read()

        # display frame
        result = process_frame(frame, type_process)
        # get info from the frame
        data = detect(result)
        # draw things in the frame
        drawed = write_frame(result, camera_object.text)

        # show the frame
        cv2.imshow(cam_name, drawed)

        # wait for user to press 'q'
        if cv2.waitKey(1) & 0xFF == ord('a'):
            camera_object.detect = True
            camera_object.text = "Camera detected!"

        if cv2.waitKey(1) & 0xFF == ord('q'):
            camera_object.event.set()
            break


def write_frame(frame, text):
    return cv2.putText(frame, text, (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)


def process_frame(frame, type_process):
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