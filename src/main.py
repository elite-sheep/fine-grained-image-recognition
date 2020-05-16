# Copyright 2020 Yuchen Wang

import cv2 as cv
import numpy as np
import pandas as pd

from features.color_histogram import ColorHistogram
from features.gamma import Gamma
from models.alexnet import AlexNet
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

    colorHist = ColorHistogram(4, 4, 16)
    gamma1 = Gamma(1.5)
    gamma2 = Gamma(0.7)
    resize = (384, 384)

    X = np.empty([row, 2*4*4*16*3])
    Y = np.zeros([row], dtype=np.int16)
    for i in range(10):
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
        X[i] /= np.linalg.norm(X[i])
        print(pathPrefix+imageId)

    return np.array(X), Y

def loadAllImages(labelFile, prefix):
    df = pd.read_csv(labelFile)
    row, col = df.shape

    resize = (224, 224)
    X = []
    Y = np.zeros([row, 3], dtype=np.float32)
    for i in range(row):
        imageId = df['image_id'][i]
        label = df['label'][i]
        y = getLabelIndex(label)
        Y[i][y] = 1.0
        image = cv.imread(prefix+imageId, 1)
        if image.shape[0] < image.shape[1]:
            image = cv.rotate(image, cv.ROTATE_90_CLOCKWISE)
        image = cv.resize(image, resize).astype(np.float32)
        image /= 255.0
        image[:,:,0] = image[:,:,0] - 0.485
        image[:,:,1] = image[:,:,1] - 0.456
        image[:,:,2] = image[:,:,2] - 0.406
        X.append(image)
        print(prefix+imageId)

    return np.array(X), Y

def main():
    trainLabelFile = '/tmp2/yucwang/data/mongo/train.csv'
    trainPrefix = '/tmp2/yucwang/data/mongo/C1-P1_Train/'
    validLabelFile = '/tmp2/yucwang/data/mongo/dev.csv'
    validPrefix = '/tmp2/yucwang/data/mongo/C1-P1_Dev/'

    trainX, trainY = loadAllImages(trainLabelFile, trainPrefix)
    testX, testY = loadAllImages(validLabelFile, validPrefix)

    validIndicies = np.random.choice(testX.shape[0], 75)
    validX = testX[validIndicies]
    validY = testY[validIndicies]

    model = AlexNet()
    model.train(weightsSavePath = './bin/exp5/', 
            batches=9000, batchSize=128, learningRate=0.1, X=trainX, 
            Y=trainY, validX=validX, validY=validY, decayStep=[3480, 5000])
    model.evaluate(testX, testY)
#
#    trainX, trainY = extractFeatures(trainLabelFile, trainPrefix)
#    validX, validY = extractFeatures(validLabelFile, validPrefix)
#
#    np.save('./train_x.npy', trainX)
#    np.save('./train_y.npy', trainY)
#    np.save('./val_x.npy', validX)
#    np.save('./val_y.npy', validY)

#    trainX = np.load('./bin/exp2/train_x.npz.npy')
#    trainY = np.load('./bin/exp2/train_y.npz.npy')
#    validX = np.load('./bin/exp2/val_x.npz.npy')
#    validY = np.load('./bin/exp2/val_y.npz.npy')
#
#    model = SVM(penalty='l2', loss='squared_hinge',
#            C=0.85, maxIter=2000)
#    print("SVM: Training get started.")
#    model.train(trainX, trainY)
#
#    print("SVM: Validation get started.")
#    acc, metrics = model.valid(validX, validY, classNum=3)
#    print(acc)
#    print(metrics)

main()
