import math

import cv2
import pandas as pd
import scipy.ndimage
import tensorflow as tf
from matplotlib import pyplot as plt
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


class ImageAlterationsClass:

    def increaseContrast(img):
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l_channel = lab[:, :, 0]
        a = lab[:, :, 1]
        b = lab[:, :, 2]

        # Applying CLAHE to L-channel
        # feel free to try different values for the limit and grid size:
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        cl = clahe.apply(l_channel)

        # merge the CLAHE enhanced L-channel with the a and b channel
        limg = cv2.merge((cl, a, b))

        # Converting image from LAB Color model to BGR color spcae
        enhanced_img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

        # Stacking the original image with the enhanced image
        # result = np.hstack((img, enhanced_img))
        # cv2.imshow('Result', cv2.resize(result, None, fx=0.5, fy=0.5))
        return enhanced_img

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
                if distance > flowThreshold and ((xmin < x < xmax) and (ymin < y < ymax)):
                    list_flow.append([x, y])
                    # cv2.arrowedLine(rgb, (x, y), (int(x + flow[0]), int(y + flow[1])), color=(0, 0, 255), thickness=2)
                else:
                    rgb[y][x] = np.array([0, 0, 0])

        return list_flow

    def getHistogramofGradientsChannels(im, xmin, xmax, ymin, ymax):
        im = im[ymin:ymax, xmin:xmax]
        img = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

        cell_size = (8, 8)  # h x w in pixels
        block_size = (2, 2)  # h x w in cells
        nbins = 9  # number of orientation bins

        # winSize is the size of the image cropped to an multiple of the cell size
        hog = cv2.HOGDescriptor(_winSize=(img.shape[1] // cell_size[1] * cell_size[1],
                                          img.shape[0] // cell_size[0] * cell_size[0]),
                                _blockSize=(block_size[1] * cell_size[1],
                                            block_size[0] * cell_size[0]),
                                _blockStride=(cell_size[1], cell_size[0]),
                                _cellSize=(cell_size[1], cell_size[0]),
                                _nbins=nbins)

        n_cells = (img.shape[0] // cell_size[0], img.shape[1] // cell_size[1])
        hog_feats = hog.compute(img) \
            .reshape(n_cells[1] - block_size[1] + 1,
                     n_cells[0] - block_size[0] + 1,
                     block_size[0], block_size[1], nbins) \
            .transpose((1, 0, 2, 3, 4))  # index blocks by rows first
        # hog_feats now contains the gradient amplitudes for each direction,
        # for each cell of its group for each group. Indexing is by rows then columns.

        gradients = np.zeros((n_cells[0], n_cells[1], nbins))

        # count cells (border cells appear less often across overlapping groups)
        cell_count = np.full((n_cells[0], n_cells[1], 1), 0, dtype=int)

        for off_y in range(block_size[0]):
            for off_x in range(block_size[1]):
                gradients[off_y:n_cells[0] - block_size[0] + off_y + 1,
                off_x:n_cells[1] - block_size[1] + off_x + 1] += \
                    hog_feats[:, :, off_y, off_x, :]
                cell_count[off_y:n_cells[0] - block_size[0] + off_y + 1,
                off_x:n_cells[1] - block_size[1] + off_x + 1] += 1

        # Average gradients
        gradients /= cell_count

        # Preview
        plt.figure()
        plt.imshow(img, cmap='gray')
        plt.show()

        bin = 5  # angle is 360 / nbins * direction
        plt.pcolor(gradients[:, :, bin])
        plt.gca().invert_yaxis()
        plt.gca().set_aspect('equal', adjustable='box')
        plt.colorbar()
        plt.show()

    # todo getAverageImageCoordinates test
    def getAverageImageCoordinates(list_flow):
        array_flow = np.array(list_flow)
        averaged = np.average(array_flow, axis=0)
        return averaged

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

    # todo getMedianImageCoordinates test
    def getMedianImageCoordinates(list_flow):
        array_flow = np.array(list_flow)
        median = np.median(array_flow, axis=0)
        return median

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

    def getMedianImageChannelsGrayscale(im):
        im = im[im != 0]
        # median values
        g_median = np.median(im)

        return g_median

    def getKNNCoordinates(img, value, xmin, xmax, ymin, ymax):
        h, w, c = img.shape

        for x in range(w):
            for y in range(h):

                dist = np.abs(x - value[0]) + np.abs(y - value[1])

                if dist < 20 or dist > 35 or not ((xmin < x < xmax) and (ymin < y < ymax)):
                    img[y][x] = np.array(0)

    def getKNNGrayscale(im, value, xmin, xmax, ymin, ymax):
        h, w = im.shape

        for x in range(w):
            for y in range(h):

                px = im[y][x]
                dist = ((np.abs(px - value)) / 255) * 100
                # print("dist: " + str(dist))

                if dist < 20 or dist > 35 or not ((xmin < x < xmax) and (ymin < y < ymax)):
                    im[y][x] = np.array(0)

    def getKNN(im, value, xmin, xmax, ymin, ymax):
        h, w, c = im.shape

        for x in range(w):
            for y in range(h):

                px = im[y][x]
                # dist = np.sqrt(np.square(px[0] - median[0]) + np.square(px[1] - median[1]) + np.square(px[2] - median[2]))
                dist = ((np.abs(px[0] - value[0]) + np.abs(px[1] - value[1]) + np.abs(px[2] - value[2])) / 765) * 100
                # print("dist: " + str(dist))

                if dist < 16 or dist > 46 or not ((xmin < x < xmax) and (ymin < y < ymax)):
                    im[y][x] = np.array([0, 0, 0])
