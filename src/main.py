# Copyright 2020 Yuchen Wang

import cv2 as cv
import numpy as np
import pandas as pd
import random as rd
import tensorflow as tf

from PIL import Image
from PIL import ImageFilter
from PIL import ImageEnhance

from features.color_histogram import ColorHistogram
from features.gamma import Gamma
from models.alexnet import AlexNet
from models.svm import SVM
from models.kernel_svm import KernelSVM

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
    filenameList = []
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
        
        filenameList.append(pathPrefix+imageId)

    return np.array(X), Y, filenameList

def loadAllImages(labelFile, prefix, argument=False):
    df = pd.read_csv(labelFile)
    row, col = df.shape

    resize = (227, 227)
    X = []
    Y = []
    filenameList = []
    gamma1 = Gamma(1.6)
    for i in range(row):
        imageId = df['image_id'][i]
        label = df['label'][i]
        y = getLabelIndex(label)
        image = cv.imread(prefix+imageId, 1)
        image = gamma1.extract(image)

        if row > col:
            image = cv.rotate(image, cv.ROTATE_90_CLOCKWISE)
        image = cv.resize(image, resize).astype(np.float64)
        image /= 255.0
        image[:,:,0] = image[:,:,0] - 0.264
        image[:,:,1] = image[:,:,1] - 0.294
        image[:,:,2] = image[:,:,2] - 0.506

        sample = rd.uniform(0.0, 1.0)
        if sample < 0.4 and argument == True:
            flipedImage = cv.flip(image, -1)
            X.append(flipedImage)
            Y.append(y)

        X.append(image)
        Y.append(y)
        filenameList.append(prefix+imageId)
        print(prefix+imageId)

    return np.array(X), tf.keras.utils.to_categorical(np.array(Y)), filenameList

def main():
    trainLabelFile = '/tmp2/yucwang/data/mongo_data/train.csv'
    trainPrefix = '/tmp2/yucwang/data/mongo_data/C1-P1_Train/'
    validLabelFile = '/tmp2/yucwang/data/mongo_data/dev.csv'
    validPrefix = '/tmp2/yucwang/data/mongo_data/C1-P1_Dev/'

    trainX, trainY, trainFiles = loadAllImages(trainLabelFile, trainPrefix, argument=True)
    testX, testY, files = loadAllImages(validLabelFile, validPrefix)

    validIndicies = np.random.choice(testX.shape[0], 75)
    validX = testX[validIndicies]
    validY = testY[validIndicies]

    print('Training with {} images.'.format(trainY.shape[0]))

    model = AlexNet(inputShape=[227, 227, 3])
    model.loadWeights('./pretrained/alexnet_weights.h5')
    model.train(weightsSavePath = './bin/exp12/', 
            batches=18000, batchSize=128, learningRate=0.001, X=trainX, 
            Y=trainY, validX=validX, validY=validY, decayStep=[6400])
    model.evaluate(testX, testY, files)
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
#    model = KernelSVM(max_iter=-1, grid_search=True)
#    print("SVM: Training get started.")
#    model.train(trainX, trainY)
#
#    print("SVM: Validation get started.")
#    acc, war, metrics = model.valid(validX, validY, classNum=3)
#    print(acc)
#    print(war)
#    print(metrics)

main()
