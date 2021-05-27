import cv2
import numpy as np

img = cv2.imread('lena.jpg')
print(img.shape)
# returns tuple of no of rows clmns , channels
print(img.dtype)
print(img.size) # no of pixels

cv2.imshow('image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()