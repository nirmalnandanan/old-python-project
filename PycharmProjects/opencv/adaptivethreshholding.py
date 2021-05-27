import cv2
import numpy as np
img = cv2.imread('sudoku.jpg', 0)

_, th1 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
th2 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 10)
th3 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 10)

cv2.imshow('image', img)
cv2.imshow('th1', th1)
cv2.imshow('adpt', th2)
cv2.imshow('gauss', th3)
cv2.waitKey(0)
cv2.destroyAllWindows()
