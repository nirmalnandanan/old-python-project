import cv2
import numpy as np
from matplotlib import pyplot as plt
img = cv2.imread('ballz.jpg', cv2.IMREAD_GRAYSCALE)
_, mask = cv2.threshold(img, 200, 220, cv2.THRESH_BINARY_INV)
kernel = np.ones((2, 2), np.uint8)
dilation = cv2.dilate(mask, kernel, iterations=10)
erosion = cv2.erode(mask, kernel, iterations= 10)
# erosion followed by dilation is opening
opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
# closed is dilation then erosion
closing = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
# kernel is shape here we use rectangle size is the 2,2
titles = ['image', 'mask', 'dilation', 'erosion', 'opening', 'closing']
image = [img, mask, dilation, erosion, opening, closing]
for i in range(6):
    plt.subplot(2, 3, i+1)
    plt.imshow(image[i], 'gray')
    plt.title(titles[i])
    plt.xticks([])
    plt.yticks([])
plt.show()
cv2.waitKey(0)
cv2.destroyAllWindows()