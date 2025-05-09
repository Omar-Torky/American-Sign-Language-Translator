import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
from tensorflow import keras
import numpy as np
import math
import time
from tensorflow.keras.layers import DepthwiseConv2D



cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
offset = 20
imgSize = 300
counter = 0
labels = ['A', 'B', 'C']

folder = "Data/C"
classifier = Classifier("Model/keras_model.h5","Model/labels.txt")


while True:
    success, img = cap.read()
    imgOutput = img.copy()
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
            prediction, index = classifier.getPrediction(imgWhite,draw=False)
            print(prediction, index)


        else:
            const = imgSize / w
            new_h = math.ceil(const * h)
            imgResize = cv2.resize(imgCrop, (imgSize, new_h))
            hGap = math.ceil((imgSize - new_h) / 2)
            imgWhite[hGap:new_h + hGap, :] = imgResize
            prediction, index = classifier.getPrediction(imgWhite, draw=False)

        cv2.putText(imgOutput, labels[index], (x,y-20), cv2.FONT_HERSHEY_COMPLEX, 2, (255,0,255), 2)
        cv2.rectangle(imgOutput, (x-offset, y-offset), (x+w+offset, y+h+offset), (255, 0, 255), 4)

        cv2.imshow("ImageCrop", imgCrop)
        cv2.imshow("ImageWhite", imgWhite)



    cv2.imshow("Image", imgOutput)
    cv2.waitKey(1)

