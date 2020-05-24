# Copyright 2020 Yuchen Wang

import os
import cv2 as cv
import numpy as np
import pandas as pd
import random as rd
import tensorflow as tf

from features.color_histogram import ColorHistogram
from features.gamma import Gamma
from PIL import Image
from PIL import ImageFilter
from PIL import ImageEnhance
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

def loadAllImages(labelFile, prefix, argument=False):
    df = pd.read_csv(labelFile)
    row, col = df.shape

    resize = (227, 227)
    X = []
    Y = []
    gamma1 = Gamma(1.5)
    for i in range(row):
        imageId = df['image_id'][i]
        label = df['label'][i]
        y = getLabelIndex(label)
        image = cv.imread(prefix+imageId, 1)
        image = gamma1.extract(image)
        image = cv.resize(image, resize).astype(np.float64)
        image /= 255.0
        image[:,:,0] = image[:,:,0] - 0.485
        image[:,:,1] = image[:,:,1] - 0.456
        image[:,:,2] = image[:,:,2] - 0.406

        sample = rd.uniform(0.0, 1.0)
        if sample < 0.3 and argument == True:
            flipedImage = cv.flip(image, -1)
            X.append(flipedImage)
            Y.append(y)

        X.append(image)
        Y.append(y)
        print(prefix+imageId)

    return np.array(X), tf.keras.utils.to_categorical(np.array(Y))

def cropCenter(image, cropWidth, cropHeight):
    width, height = image.size
    return image.crop(((width - cropWidth) // 2, (height - cropHeight) // 2,
                         (width + cropWidth) // 2, (height + cropHeight) // 2))

def cropMaxSquare(image):
    return cropCenter(image, min(image.size), min(image.size))

def imagePreprocess(labelFile, pathPrefix):
    print(labelFile)
    df = pd.read_csv(labelFile)
    row, col = df.shape
    size = 224
    pathNew = pathPrefix[:-1] + '_preprocess/'
    os.mkdir(pathNew)

    for i in range(row):
        imageId = df['image_id'][i]
        image = Image.open(pathPrefix+imageId)
        # crop
        imageNew = cropMaxSquare(image)
        imageNew = imageNew.resize((size,size),Image.ANTIALIAS)
        # enhance contrast
        imageNew = ImageEnhance.Contrast(imageNew).enhance(1.1)
        # remove white spots
        imageNew = cv.cvtColor(np.asarray(imageNew),cv.COLOR_RGB2BGR)  # PIL to cv
        imageNew = cv.fastNlMeansDenoisingColored(imageNew,None,15,10,7,21)
        # sharpen
        imageNew = Image.fromarray(cv.cvtColor(imageNew,cv.COLOR_BGR2RGB))  #cv to PIL
        imageNew = imageNew.filter(ImageFilter.SHARPEN)
        # edge detection (canny)
        #imageNew = cv.cvtColor(np.asarray(imageNew),cv.COLOR_RGB2BGR)
        #imageNew = cv.Canny(imageNew,120,130)
        #imageNew = Image.fromarray(cv.cvtColor(imageNew,cv.COLOR_BGR2RGB))
        # save image
        imageNew.save(pathNew+imageId, quality=100, subsample=0)
        print(pathNew+imageId)

    return pathNew

def main():
    filePath = '/tmp2/yucwang/data/mongo/'
    trainLabelFile = filePath + 'train.csv'
    trainPrefix = filePath + 'C1-P1_Train/'
    validLabelFile = filePath + 'dev.csv'
    validPrefix = filePath + 'C1-P1_Dev/'

#    trainPreprocess = imagePreprocess(trainLabelFile, trainPrefix)
#    validPreprocess = imagePreprocess(validLabelFile, validPrefix)
#
#    trainPreprocess = trainPrefix[:-1] + '_preprocess/'
#    validPreprocess = validPrefix[:-1] + '_preprocess/'
#    
#    trainX, trainY = extractFeatures(trainLabelFile, trainPreprocess)
#    validX, validY = extractFeatures(validLabelFile, validPreprocess)

    trainX, trainY = loadAllImages(trainLabelFile, trainPrefix, argument=True)
    testX, testY = loadAllImages(validLabelFile, validPrefix)

    np.save('./train_x.npy', trainX)
    np.save('./train_y.npy', trainY)
    np.save('./val_x.npy', validX)
    np.save('./val_y.npy', validY)

    validIndicies = np.random.choice(testX.shape[0], 75)
    validX = testX[validIndicies]
    validY = testY[validIndicies]

    print('Training with {} images.'.format(trainY.shape[0]))

    model = AlexNet(inputShape=[227, 227, 3],
            pretrainedWeights="./pretrained/alexnet_weights.h5")
    model.train(weightsSavePath = './bin/exp9/', 
            batches=6500, batchSize=128, learningRate=0.001, X=trainX, 
            Y=trainY, validX=validX, validY=validY, decayStep=[2400, 5000])
    model.evaluate(testX, testY)
#
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
