import cv2
import numpy as np
img = cv2.imread('lena.jpg')
b, g, r = cv2.split(img)
# print(b, g, r)
# ball = img[280:333, 342:233]
# img[133:232, 100:222] = ball
# cv2.add()
# cv2.resize()
cv2.imshow('img', img)
cv2.waitKey(0)
cv2.destroyAllWindows()