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

    colorHist = ColorHistogram(2, 2, 64)
    gamma1 = Gamma(1.5)
    gamma2 = Gamma(0.6)
    resize = (432, 432)

    X = np.empty([row, 2*2*2*64*3])
    Y = np.zeros([row], dtype=np.int16)
    for i in range(row):
        imageId = df['image_id'][i]
        label = df['label'][i]
        Y[i] = getLabelIndex(label)
        image = cv.imread(pathPrefix+imageId, 1)
        image1 = gamma1.extract(image)
        image1 = cv.resize(image1, resize)
        image2 = gamma2.extract(image)
        image2 = cv.resize(image2, resize)
        X[i][0:768] = colorHist.extract(image1, normalize=True)
        X[i][768:1536] = colorHist.extract(image2, normalize=True)
        print(pathPrefix+imageId)

    return X, Y


def main():
    trainLabelFile = '/tmp2/yucwang/data/mongo/train.csv'
    trainPrefix = '/tmp2/yucwang/data/mongo/C1-P1_Train/'
    validLabelFile = '/tmp2/yucwang/data/mongo/dev.csv'
    validPrefix = '/tmp2/yucwang/data/mongo/C1-P1_Dev/'

    trainX, trainY = extractFeatures(trainLabelFile, trainPrefix)
    validX, validY = extractFeatures(validLabelFile, validPrefix)

    np.save('./train_x.npy', trainX)
    np.save('./train_y.npy', trainY)
    np.save('./val_x.npy', validX)
    np.save('./val_y.npy', validY)

#    trainX = np.load('train_x.npy')
#    trainY = np.load('train_y.npy')
#    validX = np.load('val_x.npy')
#    validY = np.load('val_y.npy')

    model = SVM(penalty='l2', loss='squared_hinge',
            C=0.87, maxIter=2000)
    print("SVM: Training get started.")
    model.train(trainX, trainY)

    print("SVM: Validation get started.")
    acc, metrics = model.valid(validX, validY, classNum=3)
    print(acc)
    print(metrics)

main()
