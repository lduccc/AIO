import numpy as np
import cv2

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

kernel = np.ones((10, 10), np.float32) / 100.0
pad = 1

image = cv2.imread("./12a1.jpg",1)
image = cv2.resize(image, (1500, 1000))

gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


faces = face_cascade.detectMultiScale(gray_image, 1.1, 5, minSize = (30, 30), maxSize= (60, 60))
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 225, 0), 2)
    roi = image[y: y + h, x : x + w]

    roi = cv2.filter2D(roi, cv2.CV_8U, kernel)
    roi = cv2.filter2D(roi, cv2.CV_8U, kernel)
    roi = cv2.filter2D(roi, cv2.CV_8U, kernel)

    image[y : y + h, x : x+w] = roi

cv2.imshow('image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()