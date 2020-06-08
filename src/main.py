# Copyright 2020 Yuchen Wang

import os
import cv2 as cv
import numpy as np
import pandas as pd
import random as rd
import tensorflow as tf

#from PIL import Image
#from PIL import ImageFilter
#from PIL import ImageEnhance

from features.color_histogram import ColorHistogram
from features.gamma import Gamma
from models.alexnet import AlexNet
from models.googlenet import GoogleNet
from models.inception_resnet import InceptionResnet
from models.svm import SVM
#from models.kernel_svm import KernelSVM

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

    resize = (224, 224)
    X = []
    Y = []
    files = []
    gamma1 = Gamma(1.6)
    for i in range(row):
        imageId = df['image_id'][i]
        label = df['label'][i]
        y = getLabelIndex(label)
        image = cv.imread(prefix+imageId, 1)
        image = gamma1.extract(image)
        image = cv.resize(image, resize).astype(np.float32)
        image /= 255.0

        mean, stddev = cv.meanStdDev(image)
        image[:,:,0] = (image[:,:,0] - mean[0]) / stddev[0]
        image[:,:,1] = (image[:,:,1] - mean[1]) / stddev[1]
        image[:,:,2] = (image[:,:,2] - mean[2]) / stddev[2]

#        image[:,:,0] = image[:,:,0] - 0.264
#        image[:,:,1] = image[:,:,1] - 0.294
#        image[:,:,2] = image[:,:,2] - 0.506

        sample = rd.uniform(0.0, 1.0)
        if sample < 0.4 and argument == True:
            flipedImage = cv.flip(image, -1)
            X.append(flipedImage)
            Y.append(y)

        X.append(image)
        Y.append(y)
        files.append(prefix+imageId)
        print(prefix+imageId)

    return np.array(X), tf.keras.utils.to_categorical(np.array(Y)), files

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
    filePath = '/tmp2/yucwang/data/mongo_data/'
    trainLabelFile = filePath + 'train.csv'
    trainPrefix = filePath + 'C1-P1_Train/'
    validLabelFile = filePath + 'dev.csv'
    validPrefix = filePath + 'C1-P1_Dev/'

#    trainPreprocess = imagePreprocess(trainLabelFile, trainPrefix)
#    validPreprocess = imagePreprocess(validLabelFile, validPrefix)
##
#    trainPreprocess = trainPrefix[:-1] + '_preprocess/'
#    validPreprocess = validPrefix[:-1] + '_preprocess/'
#    
#    trainX, trainY = extractFeatures(trainLabelFile, trainPreprocess)
#    validX, validY = extractFeatures(validLabelFile, validPreprocess)

    trainX, trainY, trainFiles = loadAllImages(trainLabelFile, trainPrefix, argument=True)
    testX, testY, testFiles = loadAllImages(validLabelFile, validPrefix)

    validIndicies = np.random.choice(testX.shape[0], 160)
    validX = testX[validIndicies]
    validY = testY[validIndicies]
#
#    trainX = np.load('./bin/exp2/train_x.npz.npy')
#    trainY = np.load('./bin/exp2/train_y.npz.npy')
#    validX = np.load('./bin/exp2/test_x.npz.npy')
#    validY = np.load('./bin/exp2/test_y.npz.npy')

    print('Training with {} images.'.format(trainY.shape[0]))

    model = InceptionResnet(inputShape=[224, 224, 3])
    model.loadWeights('./bin/exp26/googlenet_1500.h5')
    model.train(weightsSavePath = './bin/exp27/', 
           batches=32000, batchSize=32, learningRate=0.045, X=trainX, 
            Y=trainY, validX=validX, validY=validY)
    model.evaluate(testX, testY, testFiles)
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
