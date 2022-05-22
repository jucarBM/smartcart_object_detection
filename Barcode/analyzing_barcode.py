import numpy as np
import imutils
import cv2
from os import listdir
from os.path import isfile, join

PATH_PRE = "Images/analysis/"
PATH_POST = "Images/post_analysis/"
allfiles = [f for f in listdir(PATH_PRE) if isfile(join(PATH_PRE, f))]

for file in allfiles:
    # open image and transform in gray
    img = cv2.imread(PATH_PRE+file)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #equ = cv2.equalizeHist(gray)
    # threshold
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(5, 5))
    equ = clahe.apply(gray)   
    cv2.imwrite(PATH_POST+file, equ)
