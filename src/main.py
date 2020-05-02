# Copyright 2020 Yuchen Wang

import cv2 as cv
import numpy as np
import pandas as pd

from features.color_histogram import ColorHistogram
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

    colorHist = ColorHistogram(2, 2, 64)
    resize = (256, 256)

    X = np.empty([row, 768])
    Y = np.zeros([row])
    for i in range(row):
        imageId = df['image_id'][i]
        label = df['label'][i]
        Y[i] = getLabelIndex(label)
        image = cv.imread(pathPrefix+imageId, 1)
        image = cv.resize(image, resize)
        X[i] = colorHist.extract(image, normalize=True)
        print(pathPrefix+imageId)

    return X, Y


def main():
    trainLabelFile = '/tmp2/yucwang/data/mongo/train.csv'
    trainPrefix = '/tmp2/yucwang/data/mongo/C1-P1_Train/'
    validLabelFile = '/tmp2/yucwang/data/mongo/dev.csv'
    validPrefix = '/tmp2/yucwang/data/mongo/C1-P1_Dev/'

    trainX, trainY = extractFeatures(trainLabelFile, trainPrefix)
    validX, validY = extractFeatures(validLabelFile, validPrefix)

    np.save('./train_x.npz', trainX)
    np.save('./train_y.npz', trainY)
    np.save('./val_x.npz', validX)
    np.save('./val_y.npz', validY)

    model = SVM(penalty='l2', loss='squared_hinge',
            C=0.4, maxIter=2000)
    print("SVM: Training get started.")
    model.train(trainX, trainY)

    print("SVM: Validation get started.")
    acc = model.valid(validX, validY)
    print(acc)

main()
