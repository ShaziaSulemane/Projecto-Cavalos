import math

import cv2
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import os

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

    print_var = False
    xmin = ymax = xmax = ymin = -1

    for score, (ymin, xmin, ymax, xmax), label in zip(pred_scores, pred_boxes, pred_labels):
        if score < 0.5:
            continue
        else:
            print_var = True

        if label == 'horse':
            score_txt = f'{100 * round(score)}%'
            xmin = int(xmin * ratio_w)
            ymax = int(ymax * ratio_h)
            xmax = int(xmax * ratio_w)
            ymin = int(ymin * ratio_h)

            img_boxes = cv2.rectangle(image, (xmin, ymax), (xmax, ymin), (0, 255, 0), 2)
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(img_boxes, label, (xmin, ymax - 10), font, 1.5, (255, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(img_boxes, score_txt, (xmax, ymax - 10), font, 1.5, (255, 0, 0), 2, cv2.LINE_AA)

    return img_boxes, xmin, ymax, xmax, ymin


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
    flows = cv2.calcOpticalFlowFarneback(prevFrame, nextFrame, flow=None, pyr_scale=0.5, poly_sigma=1.5, levels=4,
                                         winsize=20, iterations=2, poly_n=1, flags=0)
    for y in range(0, height, step):
        for x in range(0, width, step):
            flow = flows[y, x] * flowLength
            distance = math.sqrt(pow(flow[0], 2) + pow(flow[1], 2))
            distance = 100.0 * distance / higherDistance

            if distance > flowThreshold and ymin < y < ymax and xmin < x < xmax:
                cv2.arrowedLine(rgb, (x, y), (int(x + flow[0]), int(y + flow[1])), color=(0, 0, 255), thickness=2)


def main():
    # The file names follow: E - epochs, LR - learning rate, BS - batches, WH - width and the height of resized
    # images, CL - number of classes (s0 - without any class 0 tag, c0 - with class 0 tags) horse_detection(

    firstFrame = True
    model = loading_model(ROOT_DIR + '/model/')
    cap = cv2.VideoCapture(
        '/home/shazia/Documents/Projecto Cavalos/HorseID - dataset/Borboleta-620098100705605/Video Lateral/VID_20210625_100523.mp4')
    while True:
        _, frame = cap.read()
        if frame is None:
            break

        img, xmin, ymax, xmax, ymin = horse_detection(frame, model)

        if xmin == -1:
            continue

        frame = cv2.resize(frame, None, fx=0.5, fy=0.5)

        if firstFrame:
            imgPrevGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            firstFrame = False
        else:
            imgNextGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            opticalFlow(imgPrevGray, imgNextGray, frame, frame.shape[1], frame.shape[0], xmin*0.5, xmax*0.5, ymin*0.5, ymax*0.5, 20, 5, 1)

            imgPrevGray = imgNextGray.copy()

            cv2.imshow('Frame', frame)
            cv2.waitKey(1)


if __name__ == "__main__":
    main()
