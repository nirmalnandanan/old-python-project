import cv2
import numpy as np
from matplotlib import pyplot as plt
img = cv2.imread('lena.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
kernel = np.ones((5, 5), np.float32)/25
destination = cv2.filter2D(img, -1, kernel)
blur = cv2.blur(img, (5, 5));
# great at  center gaussian filter
gauss = cv2.GaussianBlur(img, (5,5), 0)
#  salt and pepper noise median filter
median = cv2.medianBlur(img, 5)
# bilateral filter if image should remain sharper but also smoother
bilateral = cv2.bilateralFilter(img, 9, 75, 75)
titles = ['image', '2d conv', 'blur', 'gblur', 'median', 'bilateral']
images = [img, destination, blur, gauss, median, bilateral]

for i in range(6):
    plt.subplot(2, 3, i+1)
    plt.imshow(images[i], 'gray')
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])

plt.show()
cv2.waitKey(0)
cv2.destroyAllWindows()