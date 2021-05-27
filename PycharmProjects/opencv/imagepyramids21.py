import cv2
import numpy
from matplotlib import pyplot as plt
# just to create a pyramid
# pyramid image subjected to repeated smoothing
# gaussian and laplacian
img = cv2.imread('lena.jpg')

# just to create a pyramid
layer = img.copy()
# creating an array and first element is the original
array = [layer]
# then use for loop
for i in range(5):
    layer = cv2.pyrDown(layer)
    array.append(layer)
    # just for numbering
    # cv2.imshow(str(1), layer)
layer = array[5]
# laplacian is just like edge detection its imcomplete here
# dont bother
cv2.imshow('lapla', layer)
lr = cv2.pyrDown(img)
hr = cv2.pyrUp(img)
cv2.imshow('image', img)
cv2.imshow('pydown', lr)
cv2.imshow('pydown', hr)
cv2.waitKey(0)
cv2.destroyAllWindows()