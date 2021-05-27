import cv2 as cv
import numpy as np


def callback(x):
    print(x)


img = np.zeros((300, 512, 3), np.uint8)
# used to create window with a name
cv.namedWindow('image')

cv.createTrackbar('B', 'image', 0, 255, callback)
cv.createTrackbar('G', 'image', 0, 255, callback)
cv.createTrackbar('R', 'image', 0, 255, callback)
switch = '0 : OFF\n 1 : ON'
cv.createTrackbar(switch, 'image', 0, 1, callback)

while 1:
    imag = cv.imshow('image', img)
    k = cv.waitKey(1)
    if k == 27:
        break
    b = cv.getTrackbarPos('B','image')
    g = cv.getTrackbarPos('G', 'image')
    r = cv.getTrackbarPos('R', 'image')
    s = cv.getTrackbarPos(switch, 'image')

    if s == 0:
        img[:] = 0
    if s == 1:
        img[:] = [b, g, r]



cv.destroyAllWindows()
