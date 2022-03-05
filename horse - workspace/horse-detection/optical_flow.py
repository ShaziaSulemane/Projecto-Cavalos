#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import math

# frame = cv2.imread('../Images/ss.jpg', cv2.IMREAD_ANYCOLOR)

width = 640
height = 520
cap = cv2.VideoCapture('/home/shazia/Documents/Projecto Cavalos/HorseID - dataset/Borboleta-620098100705605/Video Lateral/VID_20210625_100523.mp4')
firstFrame = True
step = 20  # digital number of optical flow
flowLength = 5  # digital number
higherDistance = math.sqrt(pow(width, 2) + pow(height, 2))
flowThreshold = 1.2  # percentage


def opticalFlow(prevFrame, nextFrame, rgb):

    flows = cv2.calcOpticalFlowFarneback(prevFrame, nextFrame, flow=None, pyr_scale=0.5, poly_sigma=1.5, levels=4,
                                         winsize=20, iterations=2, poly_n=1, flags=0)
    for y in range(0, height, step):
        for x in range(0, width, step):
            flow = flows[y, x] * flowLength
            distance = math.sqrt(pow(flow[0], 2) + pow(flow[1], 2))
            distance = 100.0 * distance / higherDistance

            if distance > flowThreshold:
                cv2.arrowedLine(rgb, (x, y), (int(x + flow[0]), int(y + flow[1])), color=(0, 0, 255), thickness=2)


while True:

    if not cap.isOpened():
        continue

    ret, frame = cap.read()

    if frame is None:
        break

    frame = cv2.resize(frame, None, fx=0.5, fy=0.5)

    if firstFrame:
        imgPrevGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        firstFrame = False
    else:
        imgNextGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        opticalFlow(imgPrevGray, imgNextGray, frame)

        imgPrevGray = imgNextGray.copy()

        cv2.imshow('Frame', frame)
        cv2.waitKey(1)

cap.release()
cv2.destroyAllWindows()
