import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
offset = 20
imgSize = 300
counter = 0

folder = "Data/D"

while True:
    success, img = cap.read()
    hands, img = detector.findHands(img)
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']
        imgWhite = np.ones((imgSize,imgSize,3), np.uint8)*255
        imgCrop = img[y-offset:y + h+offset, x-offset:x + w+offset]


        Ratio = h/w

        if Ratio > 1:
            const = imgSize/h
            new_w = math.ceil(const*w)
            imgResize = cv2.resize(imgCrop, (new_w, imgSize))
            wGap = math.ceil((imgSize-new_w)/2)
            imgWhite[:, wGap:new_w + wGap] = imgResize

        else:
            const = imgSize / w
            new_h = math.ceil(const * h)
            imgResize = cv2.resize(imgCrop, (imgSize, new_h))
            hGap = math.ceil((imgSize - new_h) / 2)
            imgWhite[hGap:new_h + hGap, :] = imgResize

        cv2.imshow("ImageCrop", imgCrop)
        cv2.imshow("ImageWhite", imgWhite)



    cv2.imshow("Image", img)
    key = cv2.waitKey(1)
    if key == ord("s"):
        counter += 1
        cv2.imwrite(f'{folder}/Image_{time.time()}.jpg', imgWhite)
        print(counter)
