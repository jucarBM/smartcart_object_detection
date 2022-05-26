import cv2
fgbg = cv2.createBackgroundSubtractorMOG2(200, 200, detectShadows=True)


def get_background(frame, n=8):
    # gray image
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # equalize the histogram
    equ = cv2.equalizeHist(gray)
    fgmask = fgbg.apply(equ)
    # apply canny edge detection
    canny = cv2.Canny(fgmask, 50, 150)
    # and canny
    and_canny = cv2.bitwise_and(canny, fgmask)
    # thesholding
    ret, thresh = cv2.threshold(and_canny, 127, 255, 0)

    return thresh