

import rawpy
import imageio
import cv2

path = 'IMG_20220716_152419_1.dng'
with rawpy.imread(path) as raw:
    rgb = raw.postprocess()

resizedFrame = cv2.resize(rgb, (0, 0), fx=0.20, fy=0.20)
BGRR = cv2.cvtColor(resizedFrame, cv2.COLOR_RGB2BGR)
import cv2
cv2.imshow('', BGRR)
cv2.waitKey()
cv2.destroyAllWindows()
