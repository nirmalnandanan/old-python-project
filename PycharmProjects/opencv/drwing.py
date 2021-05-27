import cv2
import numpy as np

#img = cv2.imread('lena.jpg', 1)
# numpy zeros mtd
# hgt wdth 3 dtype    gives us black image
img = np.zeros([512, 512, 3], np.uint8)
# name strtng end bgr thickness
img = cv2.line(img, (0, 0), (255, 255), (255, 0, 0), 5)
img = cv2.arrowedLine(img, (0, 50), (255, 255), (0, 150, 0), 5)
# img = cv2.circle()
# img = cv2.rectangle()
font = cv2.FONT_HERSHEY_COMPLEX
img = cv2.putText(img,'this is lena', (10,50), font, 4, (0,255,255), 10,cv2.LINE_AA )
cv2.imshow('image', img)
k = cv2.waitKey(0)
cv2.destroyAllWindows()