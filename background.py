import cv2
import cvzone
from cvzone.SelfiSegmentationModule import SelfiSegmentation
import os

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)
segmentor = SelfiSegmentation()

while True:
    success, img = cap.read()
    i = segmentor.removeBG(img, (255, 0, 255))

    cv2.imshow("Image", img)
    cv2.imshow("Image Out", i)
    cv2.waitKey(1)

