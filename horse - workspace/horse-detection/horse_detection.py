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
    #loading the model
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


def horse_detection(image_path, model, image_index):
    width = 1028
    height = 1028

    # Load image by Opencv2
    image = cv2.imread(image_path)

    # Resize to respect the input_shape
    inp = cv2.resize(image, (width, height))

    # Convert img to RGB

    # Create a basic model instance
    #model = create_model()

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
    pred_labels = [labels[i-1] for i in pred_labels]
    pred_boxes = boxes.numpy()[0].astype('int')
    pred_scores = scores.numpy()[0]

    print_var = False

    for score, (ymin, xmin, ymax, xmax), label in zip(pred_scores, pred_boxes, pred_labels):
        if score < 0.5:
            continue
        else:
            print_var = True

        score_txt = f'{100 * round(score)}%'
        img_boxes = cv2.rectangle(inp, (xmin, ymax), (xmax, ymin), (0, 255, 0), 2)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img_boxes, label, (xmin, ymax - 10), font, 1.5, (255, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(img_boxes, score_txt, (xmax, ymax - 10), font, 1.5, (255, 0, 0), 2, cv2.LINE_AA)

    cv2.imshow('result', img_boxes)
    cv2.waitKey(0)

    if print_var:
        cv2.imwrite('result/result' + str(image_index), img_boxes)


def runAllImages(path):
    model = loading_model(ROOT_DIR + '/model/')
    for root, dirs, files in os.walk(os.path.join(path)):
        files = sorted(files)
        while len(files):
            file = files[0]
            horse_detection(os.path.join(root, file), model, file)
            del files[0]


def main():
    # The file names follow: E - epochs, LR - learning rate, BS - batches, WH - width and the height of resized
    # images, CL - number of classes (s0 - without any class 0 tag, c0 - with class 0 tags) horse_detection(
    # '/home/rafael/horse_images/Frames/D-Bacatum-1-62.jpg', 'models/detection_model') testing_model(
    # 'data/Frames_data_14.csv', 'models/detection_model')
    runAllImages(ROOT_DIR + '/data/')
    # horse_detection('data/4.png', model, 4)


if __name__ == "__main__":
    main()
