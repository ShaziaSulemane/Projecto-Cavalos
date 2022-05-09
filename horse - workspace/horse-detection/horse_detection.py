import math

import cv2
import pandas as pd
import scipy.ndimage
import tensorflow as tf
from tensorflow import keras
import os
import numpy as np

import PIL
from PIL import Image
import requests
from io import BytesIO
from PIL import ImageFilter
from PIL import ImageEnhance
from IPython.display import display

ROOT_DIR = os.path.dirname(os.path.realpath(__file__))


def loading_model(model_path):
    new_model = tf.keras.models.load_model(model_path)
    return new_model


def testing_model(csv_filename, model_path):
    # loading the model
    model = loading_model(model_path)

    # get the image paths
    images_dataframe = pd.read_csv(csv_filename)
    np_images_dataframe = images_dataframe.to_numpy()
    # initialize the data and labels
    print("[INFO] loading images...")

    # loop over the input images
    i = 0
    for frame_element in np_images_dataframe:
        i = i + 1
        if i % 1 == 0:
            print(i)
        # load the image, pre-process it, and store it in the data list
        result, certenty = horse_detection(frame_element[1], model, i)
        print(result)


# Define a simple sequential model
def create_model():
    model = tf.keras.models.Sequential([
        keras.layers.Dense(512, activation='relu', input_shape=(784,)),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(10)
    ])

    model.compile(optimizer='adam',
                  loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=[tf.metrics.SparseCategoricalAccuracy()])

    return model


def horse_detection(image, model):
    width = 1028
    height = 1028

    # Load image by Opencv2
    # image = cv2.imread(image_path)

    # Resize to respect the input_shape
    inp = cv2.resize(image, (width, height))
    img_boxes = image

    ratio_h = image.shape[0] / height
    ratio_w = image.shape[1] / width

    # Convert img to RGB

    # Create a basic model instance
    # model = create_model()

    # Converting to uint8
    rgb_tensor = tf.convert_to_tensor(inp, dtype=tf.uint8)

    # Add dims to rgb_tensor
    rgb_tensor = tf.expand_dims(rgb_tensor, 0)

    # Loading model
    new_model = model

    # Creating prediction
    boxes, scores, classes, num_detections = new_model(rgb_tensor)

    # Loading csv with labels of classes
    labels = pd.read_csv('coco-labels-paper.txt', header=None)
    # labels = labels['OBJECT (2017 REL.)']
    labels = labels.to_numpy()[:, 0]

    # Processing outputs
    pred_labels = classes.numpy().astype('int')[0]
    pred_labels = [labels[i - 1] for i in pred_labels]
    pred_boxes = boxes.numpy()[0].astype('int')
    pred_scores = scores.numpy()[0]
    new_pred_boxes = []

    print_var = False
    i = 0
    for score, (ymin, xmin, ymax, xmax), label in zip(pred_scores, pred_boxes, pred_labels):
        if score < 0.5:
            continue

        if label == 'horse':
            score_txt = f'{100 * round(score)}%'
            xmin = int(xmin * ratio_w)
            ymax = int(ymax * ratio_h)
            xmax = int(xmax * ratio_w)
            ymin = int(ymin * ratio_h)

            new_pred_boxes.append([ymin, xmin, ymax, xmax])

            img_boxes = cv2.rectangle(image, (xmin, ymax), (xmax, ymin), (0, 255, 0), 2)
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(img_boxes, label, (xmin, ymax - 10), font, 1.5, (255, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(img_boxes, score_txt, (xmax, ymax - 10), font, 1.5, (255, 0, 0), 2, cv2.LINE_AA)

    return img_boxes, new_pred_boxes


def runAllImages(path):
    model = loading_model(ROOT_DIR + '/model/')
    for root, dirs, files in os.walk(os.path.join(path)):
        files = sorted(files)
        while len(files):
            file = files[0]
            horse_detection(os.path.join(root, file), model, file)
            del files[0]


def opticalFlow(prevFrame, nextFrame, rgb, width, height, xmin, xmax, ymin, ymax, step, flowLength, flowThreshold):
    higherDistance = math.sqrt(pow(width, 2) + pow(height, 2))
    # opticalFlowGPU = cv2.cuda_FarnebackOpticalFlow.create(4, 0.5, False, 20, 2, 5, 1.5, 0)
    # flows = opticalFlowGPU.calc(cv2.cuda_GpuMat(prevFrame), cv2.cuda_GpuMat(nextFrame), None).download()
    flows = cv2.calcOpticalFlowFarneback(prevFrame, nextFrame, flow=None, pyr_scale=0.5, poly_sigma=1.5, levels=4,
                                         winsize=20, iterations=2, poly_n=1, flags=0)
    list_flow = []
    for y in range(0, height, step):
        for x in range(0, width, step):
            flow = flows[y, x] * flowLength
            distance = math.sqrt(pow(flow[0], 2) + pow(flow[1], 2))
            distance = 100.0 * distance / higherDistance

            if distance > flowThreshold and (xmin < x < xmax) and (ymin < y < ymax):
                list_flow.append([x, y])
                # cv2.arrowedLine(rgb, (x, y), (int(x + flow[0]), int(y + flow[1])), color=(0, 0, 255), thickness=2)
            else:
                rgb[y][x] = np.array([0, 0, 0])

    return list_flow

def getHistogramofGradientsChannels (img, xmin, xmax, ymin, ymax):
    # crop image
    # Python gradient calculation
    # Read image
    im = cv2.imread('bolt.png')
    im = np.float32(im) / 255.0

    # Calculate gradient
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=1)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=1)

    mag, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)


def getAverageImageChannels(img):
    b, g, r = cv2.split(img)  # Split channels
    # Remove zeros
    b = b[b != 0]
    g = g[g != 0]
    r = r[r != 0]

    b_average = np.average(b)
    g_average = np.average(g)
    r_average = np.average(r)

    return r_average, g_average, b_average


def getMedianImageChannels(im):
    b, g, r = cv2.split(im)  # Split channels
    # Remove zeros
    b = b[b != 0]
    g = g[g != 0]
    r = r[r != 0]
    # median values
    b_median = np.median(b)
    r_median = np.median(r)
    g_median = np.median(g)

    return r_median, g_median, b_median


def getKNN(im, value, xmin, xmax, ymin, ymax):
    h, w, c = im.shape

    for x in range(w):
        for y in range(h):

            px = im[y][x]
            # dist = np.sqrt(np.square(px[0] - median[0]) + np.square(px[1] - median[1]) + np.square(px[2] - median[2]))
            dist = np.abs(px[0] - value[0]) + np.abs(px[1] - value[1]) + np.abs(px[2] - value[2])

            if dist < 125 or dist > 350 or not ((xmin < x < xmax) and (ymin < y < ymax)):
                im[y][x] = np.array([0, 0, 0])


def main():
    # The file names follow: E - epochs, LR - learning rate, BS - batches, WH - width and the height of resized
    # images, CL - number of classes (s0 - without any class 0 tag, c0 - with class 0 tags) horse_detection(

    # find color of horse
    # find other pixels with 'same' color

    firstFrame = True
    model = loading_model(ROOT_DIR + '/model/')
    cap = cv2.VideoCapture(
        '/home/shazia/Documents/Projecto Cavalos/HorseID - dataset/Borboleta-620098100705605/Video Lateral/VID_20210625_100523.mp4')
    while True:
        _, frame = cap.read()
        if frame is None:
            break

        flow_img = frame.copy()
        img, pred_boxes = horse_detection(frame, model)
        # print("box: " + str(pred_boxes))

        if len(pred_boxes):

            if firstFrame:
                imgPrevGray = cv2.cvtColor(flow_img, cv2.COLOR_BGR2GRAY)
                firstFrame = False
            else:
                imgNextGray = cv2.cvtColor(flow_img, cv2.COLOR_BGR2GRAY)

                for (ymin, xmin, ymax, xmax) in pred_boxes:
                    list_flow = opticalFlow(imgPrevGray, imgNextGray, flow_img, flow_img.shape[1], flow_img.shape[0],
                                            xmin, xmax, ymin, ymax, 1, 5, 1)

                imgPrevGray = imgNextGray.copy()

                r_median, g_median, b_median = getMedianImageChannels(flow_img)
                getKNN(img, [b_median, g_median, r_median], xmin, xmax, ymin, ymax)
                print(r_median, g_median, b_median)

        # stacked = np.hstack((frame, img))
        cv2.imshow('Frame', cv2.resize(img, None, fx=0.75, fy=0.75))
        cv2.waitKey(1)


if __name__ == "__main__":
    main()
