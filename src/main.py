# Copyright 2020 Yuchen Wang

import cv2 as cv
import numpy as np
import pandas as pd

from features.color_histogram import ColorHistogram
from features.gamma import Gamma
from models.svm import SVM

def getLabelIndex(label):
    if label == 'A':
        return 0
    elif label == 'B':
        return 1
    else:
        return 2

def extractFeatures(labelFile, pathPrefix):
    print(labelFile)
    df = pd.read_csv(labelFile)
    row, col = df.shape

    colorHist = ColorHistogram(4, 4, 32)
    gamma = Gamma(0.4)
    resize = (512, 512)

    X = np.empty([row, 4*4*32*3])
    Y = np.zeros([row], dtype=np.int16)
    for i in range(row):
        imageId = df['image_id'][i]
        label = df['label'][i]
        Y[i] = getLabelIndex(label)
        image = cv.imread(pathPrefix+imageId, 1)
        image = gamma.extract(image)
        width, height = image.shape
        image = cv.resize(image[16:width-16, 16:height-16], resize)
        X[i] = colorHist.extract(image, normalize=True)
        print(pathPrefix+imageId)

    return X, Y


def main():
#    trainLabelFile = '/tmp2/yucwang/data/mongo/train.csv'
#    trainPrefix = '/tmp2/yucwang/data/mongo/C1-P1_Train/'
#    validLabelFile = '/tmp2/yucwang/data/mongo/dev.csv'
#    validPrefix = '/tmp2/yucwang/data/mongo/C1-P1_Dev/'
#
#    trainX, trainY = extractFeatures(trainLabelFile, trainPrefix)
#    validX, validY = extractFeatures(validLabelFile, validPrefix)
#
#    np.save('./train_x.npz', trainX)
#    np.save('./train_y.npz', trainY)
#    np.save('./val_x.npz', validX)
#    np.save('./val_y.npz', validY)

    trainX = np.load('train_x.npz.npy')
    trainY = np.load('train_y.npz.npy')
    validX = np.load('val_x.npz.npy')
    validY = np.load('val_y.npz.npy')

    model = SVM(penalty='l2', loss='squared_hinge',
            C=0.85, maxIter=2000)
    print("SVM: Training get started.")
    model.train(trainX, trainY)

    print("SVM: Validation get started.")
    acc = model.valid(validX, validY)
    print(acc)

main()
