import numpy as np
import cv2

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
capture = cv2.VideoCapture(0)

kernel = np.ones((10, 10), np.float32) / 100.0
pad = 1

while True:
    _, img = capture.read()
    if img is None:
        break

    img = cv2.resize(img, (1080, 720))
    img2 = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.15, 3, minSize = (125, 125))

    for (x, y, w, h) in faces:
        height, width = img.shape[0], img.shape[1]

        x1 = max(x - pad, 0)
        y1 = max(y - pad, 0)

        x2 = min(x + w + pad, width)
        y2 = min(y + h + pad, height)
        cv2.rectangle(img, (x1 - 10, y1 - 10), (x2 + 10, y2 + 10), (0, 225, 0), 2)


        #This is for blurring after finding the face
        # roi = img[y1: y2, x1: x2]
        # roi = cv2.filter2D(roi, cv2.CV_8U, kernel)
        # roi = cv2.filter2D(roi, cv2.CV_8U, kernel)
        # roi = cv2.filter2D(roi, cv2.CV_8U, kernel)

        # img[y1: y2, x1: x2] = roi

        # cv2.putText(img, "Can't you see my face?", (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 2, cv2.LINE_AA)

    cv2.imshow('img', img)
    cv2.imshow('img2', gray)

    key = cv2.waitKey(10) & 0xff
    if key == 27:
        break

capture.release()
cv2.destroyAllWindows()