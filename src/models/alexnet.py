# Copyright 2020 Yuchen Wong

import numpy as np
import tensorflow as tf

class AlexNet(object):
    def __init__(self, 
            inputShape=[224, 224, 3],
            numClass=3,
            dropOutProb=0.5,
            pretrainedWeights=None):
        self._inputShape = inputShape
        self._numClass = numClass
        self._dropOutProb = dropOutProb
        self._pretrainedWeight = pretrainedWeights

        self._model = self.buildUpModel()

    def buildUpModel(self):

        model = tf.keras.Sequential()

        # 1st layer
        model.add(tf.keras.layers.Conv2D(input_shape=self._inputShape, 
            filters=96, kernel_size=[11, 11], strides=[4, 4],
            activation='relu', padding='same'))
        model.add(maxPool(3, 3, 2, 2, padding='VALID', name='pool1'))
        model.add(tf.keras.layers.BatchNormalization())

        # 2nd layer
        model.add(conv(5, 5, 256, 1, 1, name='conv2'))
        model.add(maxPool(3, 3, 2, 2, padding='VALID', name='pool2'))
        model.add(tf.keras.layers.BatchNormalization())

        # 3rd layer
        model.add(conv(3, 3, 384, 1, 1, name='conv3'))

        # 4th layer
        model.add(conv(3, 3, 256, 1, 1, name='conv4'))

        # 5th layer
        model.add(conv(3, 3, 256, 1, 1, name='conv5'))
        model.add(maxPool(3, 3, 2, 2, padding='VALID', name='pool5'))

        # 6th layer
        model.add(flatten())
        model.add(fullConnect(6*6*256, 4096, name='fc6'))
        model.add(dropOut(1.0 - self._dropOutProb))

        # 7th layer
        model.add(fullConnect(4096, 4096, name='fc7'))
        model.add(dropOut(1.0 - self._dropOutProb))

        # 8th layer
        model.add(fullConnect(4096, self._numClass, activation=None, name='fc8'))

        return model

    def train(self, X, Y, validX, validY, 
            weightsSavePath,
            batches = 20000,
            batchSize = 128, 
            learningRate=0.01, 
            decayStep=[1000]):

        self._optimizer = tf.keras.optimizers.Adam(learning_rate=learningRate)
        self._loss = tf.keras.losses.CategoricalCrossentropy()
        self._model.compile(optimizer=self._optimizer, loss=self._loss,
                metrics=[tf.keras.metrics.categorical_accuracy])
        self._model.build()
        self._model.summary()

        curDecayStep = 0

        for i in range(batches):
            batchIndicies = np.random.choice(X.shape[0], batchSize)
            curX = X[batchIndicies]
            curY = Y[batchIndicies]
            trainResult = self._model.train_on_batch(curX, curY)

            tf.print("Batch: ", i)
            print("TrainResult: ", dict(zip(self._model.metrics_names, trainResult)))

            if i % 10 == 0:
                self._model.evaluate(validX, validY)

            if i % 50 == 0:
                filename = weightsSavePath + "alexnet_" + str(i)
                self._model.save_weights(filename)

            if curDecayStep < len(decayStep) and i == decayStep[curDecayStep]:
                self._model.optimizer.lr.assign(self._model.optimizer.lr * 0.5)

    def evaluate(self, X, Y):
        row = X.shape[0]


        predictions = self._model.predict(X)
        acc = 0.0
        metrics = np.zeros((self._numClass, self._numClass), np.int32)
        for i in range(row):
            predict = predictions[i]
            maxPred = tf.math.argmax(predict)
            actual = tf.math.argmax(Y[i])

            metrics[int(actual)][int(maxPred)] += 1
            if int(maxPred) == int(actual):
                acc += 1.0

        print("accuracy: ", acc/row)
        print("metrics: ", metrics)
            

def conv(kernelHeight, kernelWidth, filters, strideY, strideX, padding='SAME', name=None):
    return tf.keras.layers.Conv2D(filters=filters, kernel_size=[kernelHeight, kernelWidth],
            strides=[strideY, strideX], padding=padding, activation='relu')

def dropOut(keepProb):
    return tf.keras.layers.Dropout(keepProb)

def flatten():
    return tf.keras.layers.Flatten()

def fullConnect(numIn, numOut, name, activation='relu'):
    return tf.keras.layers.Dense(numOut, input_shape=(numIn,),
            activation=activation)

def maxPool(kernelHeight, kernelWidth,
        strideY, strideX, padding='SAME', name=None):
    return tf.keras.layers.MaxPool2D(pool_size=(kernelHeight, kernelWidth),
            strides=(strideY, strideX), padding=padding)
