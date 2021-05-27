import cv2
import numpy as np


def onclick_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        blue = img[x, y, 0]
        green = img[x, y, 1]
        red = img[x, y, 2]
        cv2.circle(img, (x, y), 3, (255, 0, 0), -1)
        window = np.zeros((512, 512, 3), np.uint8)
        window[:] = [blue, green, red]
        cv2.imshow('new_window', window)
        # font = cv2.FONT_HERSHEY_COMPLEX
        # cv2.putText(window, text, (x, y), font, 3, (255, 255, 0), 2)


img = cv2.imread('lena.jpg')
cv2.imshow('image', img)
cv2.setMouseCallback('image', onclick_event)
cv2.waitKey(0)
cv2.destroyAllWindows()
