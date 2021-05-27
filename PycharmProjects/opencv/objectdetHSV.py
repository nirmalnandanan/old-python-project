import cv2
import numpy as np


def callback(x):
    pass

cap = cv2.VideoCapture(0)
cv2.namedWindow('tracking')
cv2.createTrackbar('LH', 'tracking', 0, 255, callback)
cv2.createTrackbar('LS', 'tracking', 0, 255, callback)
cv2.createTrackbar('LV', 'tracking', 0, 255, callback)
cv2.createTrackbar('UH', 'tracking', 255, 255, callback)
cv2.createTrackbar('US', 'tracking', 255, 255, callback)
cv2.createTrackbar('UV', 'tracking', 255, 255, callback)
while True:
    # frame = cv2.imread('ballz.jpg')
    _, frame = cap.read()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lh = cv2.getTrackbarPos('LH', 'tracking')
    ls = cv2.getTrackbarPos('LS', 'tracking')
    lv = cv2.getTrackbarPos('LV', 'tracking')
    uh = cv2.getTrackbarPos('UH', 'tracking')
    us = cv2.getTrackbarPos('US', 'tracking')
    uv = cv2.getTrackbarPos('UV', 'tracking')

    # threshold the hsv image for green
    # lowergreen = np.array([65, 60, 60])
    # uppergreen = np.array([80, 255, 255])
    lowergreen = np.array([lh, ls, lv])
    uppergreen = np.array([uh, us, uv])
    mask = cv2.inRange(hsv, lowergreen, uppergreen)
    res = cv2.bitwise_and(frame, frame, mask=mask)
    cv2.imshow('frame', frame)
    cv2.imshow('mask', mask)
    cv2.imshow('res', res)

    key = cv2.waitKey(1)
    if key == 27:
        break
cap.release()
cv2.destroyAllWindows()
