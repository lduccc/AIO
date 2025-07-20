import numpy as np
import cv2

img1 = cv2.imread("./mrbean_noise.jpg")
img2 = cv2.medianBlur(img1, 3)

cv2.imshow('img1', img1)
cv2.imshow('img2', img2)

cv2.waitKey(0)
cv2.destroyAllWindows()