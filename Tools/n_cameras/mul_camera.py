import cv2
from threading import Thread
# import numpy as np


class Camera(Thread):
    def __init__(self, cameraID, cam_name, type_process="normal"):
        super().__init__()
        self.cameraID = cameraID
        self.cam_name = cam_name
        self.type_process = type_process

    def run(self):
        run_camera(self.cameraID, self.cam_name, self.type_process)


def run_camera(cameraID, cam_name, type_process) -> None:
    #  open camera using opencv
    cap = cv2.VideoCapture(cameraID)

    # loop camera until user presses 'q'
    while True:
        # read frame
        _, frame = cap.read()

        # display frame
        result = process_frame(frame, type_process)
        cv2.imshow(cam_name, result)

        # wait for user to press 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


def process_frame(frame, type_process):
    if type_process == "normal":
        return frame
    elif type_process == "gray":
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    else:
        # convert frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        equ  = cv2.equalizeHist(gray)

        # apply gaussian blur
        blur = cv2.GaussianBlur(equ, (5, 5), 0)

        # apply canny edge detection
        canny = cv2.Canny(blur, 50, 150)

    # return the result
        return canny