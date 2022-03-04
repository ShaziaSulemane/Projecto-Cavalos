import cv2
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import os
import numpy as np
import matplotlib.pyplot as plt

ROOT_DIR = os.path.dirname(os.path.realpath(__file__))

cap = cv2.VideoCapture(
    '/home/shazia/Documents/Projecto Cavalos/HorseID - dataset/Bacatum 06.07.2021/Videos/Lat Esquerda.mp4')
background_object = cv2.createBackgroundSubtractorKNN(dist2Threshold=600, detectShadows=True)

kernel = None
i = 0
cx = cy = 0
num_img = 0

if not cap.isOpened():
    print("Error opening video stream or file")

while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        # blur background using HSV
        frame_h = frame.shape[0]
        frame_w = frame.shape[1]
        img_save = frame.copy()

        # Blur background using HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        background_blur_mask = cv2.inRange(hsv, (0, 75, 40), (180, 255, 255))
        mask_3d = np.repeat(background_blur_mask[:, :, np.newaxis], 3, axis=2)
        blurred_frame = cv2.GaussianBlur(frame, (25, 25), 0)
        frame = np.where(mask_3d == (255, 255, 255), frame, blurred_frame)

        foreground_mask = background_object.apply(frame)
        _, foreground_mask = cv2.threshold(foreground_mask, type=cv2.THRESH_OTSU, maxval=255, thresh=150)

        while i < 12:
            foreground_mask = cv2.erode(foreground_mask, kernel, iterations=1)
            foreground_mask = cv2.dilate(foreground_mask, kernel, iterations=2)
            i += 1

        contours, _ = cv2.findContours(foreground_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        frameCopy = frame.copy()
        cnt = max(contours, key=cv2.contourArea)

        if len(contours) != 0 and cv2.contourArea(cnt) > 93354:
            x, y, w, h = cv2.boundingRect(cnt)
            cx = int(x + w / 2)
            cy = int(y + h / 2)

            cv2.rectangle(frameCopy, (x, y), (x + w, y + h), (0, 255, 0), 2)
            # TODO clean all 500s and substitute by a constant
            inside_left_border = (cx - 500 >= 0)
            inside_right_border = (cx + 500 <= frame_w)

            inside_up_border = (cy - 500 >= 0)
            inside_down_border = (cy + 500 <= frame_h)

            if not inside_left_border:
                if not inside_up_border:
                    # up left corner
                    cx = cx + abs(cx - 500)
                    cy = cy + abs(cy - 500)
                elif not inside_down_border:
                    # down left corner
                    cx = cx + abs(cx - 500)
                    cy = cy - abs(frame_h - (cy + 500))
                else:
                    # just left border
                    cx = cx + abs(cx - 500)
            elif not inside_right_border:
                if not inside_up_border:
                    # up right corner
                    cy = cy + abs(cy - 500)
                    cx = cx - abs(frame_w - (cx + 500))
                elif not inside_down_border:
                    # down right corner
                    cy = cy - abs(frame_h - (cy + 500))
                    cx = cx - abs(frame_w - (cx + 500))
                else:
                    # just right border
                    cx = cx - abs(frame_w - (cx + 500))
            elif not inside_up_border:
                # just up border
                cy = cy + abs(cy - 500)
            elif not inside_down_border:
                # just down border
                cy = cy - abs(frame_h - (cy + 500))

            new_coord = [cx - 500, cy - 500, cx + 500,  cy + 500]
            cv2.rectangle(frameCopy, (cx - 500, cy - 500), (cx + 500, cy + 500), (255, 255, 0), 2)

            img_save = img_save[(cy - 500):(cy + 500), (cx - 500):(cx + 500)]
            if img_save.size:
                img_save = cv2.resize(img_save, (500, 500))
                cv2.imwrite("/home/shazia/Documents/Projecto Cavalos/images/image_" + str(num_img) + ".png", img_save)

                # TODO change original rectangle size to fit in 512 x 512 image
                # TODO re process new image?

                # f = open("/home/shazia/Documents/Projecto Cavalos/images/image_" + str(num_img) + ".txt", "w+")
                # f.write(str(x) + " " + str(y) + " " + str(new_w) + " " + str(new_h))
                # f.close()

                # TODO save file with coordinates in txt or json
                # num_img += 1
                # img_save_new_square = img_save.copy()
                # cv2.rectangle(img_save_new_square, (x, y), (x + new_w, y + new_h), (255, 0, 255), 2)
                cv2.imshow("Cropped and Resized", img_save)

        foreground_part = cv2.bitwise_and(frame, frame, mask=foreground_mask)
        stacked = np.hstack((frame, foreground_part, frameCopy))

        cv2.imshow('Original Frame, Extracted Foreground and Detected Horse',
                   cv2.resize(stacked, None, fx=0.33, fy=0.33))
        cv2.imshow('Clean Mask', cv2.resize(foreground_mask, None, fx=0.33, fy=0.33))

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()
