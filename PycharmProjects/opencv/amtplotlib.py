import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('lena.jpg', -1)
cv2.imshow('lena', img)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(img)
# hide the ticks x and y coordinate
plt.xticks([]), plt.yticks([])
plt.show()
cv2.waitKey(0)
cv2.destroyAllWindows()
