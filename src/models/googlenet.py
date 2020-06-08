# Copyright 2020 Yuchen Wong

import numpy as np
import tensorflow as tf

class ConvLN(tf.keras.layers.Layer):
    def __init__(self, kernelY, kernelX, filters, stridesY, strideX, **kwrags):
        super().__init__(**kwrags)

        self._conv = conv(kernelY, kernelX, filters, stridesY, strideX, activation=None)
        self._activation = tf.keras.layers.LeakyReLU(alpha=0.1)

    def call(self, inputs):
        res = self._conv(inputs)
        res = self._activation(res)

        return res


class Inception(tf.keras.layers.Layer):
    def __init__(self, params, concatAxis=-1, **kwrags):
        super().__init__(**kwrags)

        self._branch0 = params[0]
        self._branch1 = params[1]
        self._branch2 = params[2]
        self._branch3 = params[3]

        self._concatAxis = concatAxis

        self._pathway0Conv = ConvLN(1, 1, self._branch0[0], 1, 1)

        self._pathway1Conv1 = ConvLN(1, 1, self._branch1[0], 1, 1)
        self._pathway1Conv2 = ConvLN(3, 3, self._branch1[1], 1, 1)

        self._pathway2Conv1 = ConvLN(1, 1, self._branch2[0], 1, 1)
        self._pathway2Conv2 = ConvLN(1, 5, self._branch2[1], 1, 1) 
        self._pathway2Conv3 = ConvLN(5, 1, self._branch2[1], 1, 1)
                

        self._pathway3Conv1 = ConvLN(1, 3, self._branch3[0], 1, 1)
        self._pathway3Conv2 = ConvLN(3, 1, self._branch3[0], 1, 1)

    def call(self, inputs):
        pathway0 = self._pathway0Conv(inputs)

        pathway1 = self._pathway1Conv1(inputs)
        pathway1 = self._pathway1Conv2(pathway1)

        pathway2 = self._pathway2Conv1(inputs)
        pathway2 = self._pathway2Conv2(pathway2)
        pathway2 = self._pathway2Conv3(pathway2)

        pathway3 = self._pathway3Conv1(inputs)
        pathway3 = self._pathway3Conv2(pathway3)

        return tf.keras.layers.concatenate([pathway0, pathway1, pathway2, pathway3],
                axis=self._concatAxis)

class GoogleNet(object):
    def __init__(self, 
            inputShape=[224, 224, 3],
            numClass=3,
            dropOutProb=0.2,
            pretrainedWeights=None):
        self._inputShape = inputShape
        self._numClass = numClass
        self._dropOutProb = dropOutProb
        self._pretrainedWeight = pretrainedWeights

        self._model = self.buildUpModel()

    def buildUpModel(self):
        model = tf.keras.Sequential()

        # Input Layer
        model.add(tf.keras.layers.InputLayer(input_shape=self._inputShape))

        # 1st layer
        model.add(conv(7, 7, 64, 2, 2, activation=None, name='conv1'))
        model.add(tf.keras.layers.LayerNormalization())
        model.add(tf.keras.layers.LeakyReLU(alpha=0.1))
        model.add(maxPool(3, 3, 2, 2, padding='same', name='pool1'))

        # 2nd layer
        model.add(conv(1, 1, 64, 1, 1, activation=None, padding='same', name='conv2'))
        model.add(tf.keras.layers.LayerNormalization())
        model.add(tf.keras.layers.LeakyReLU(alpha=0.1))

        # 3rd layer
        model.add(conv(3, 3, 192, 1, 1, activation=None, padding='same', name='conv3'))
        model.add(tf.keras.layers.LayerNormalization())
        model.add(tf.keras.layers.LeakyReLU(alpha=0.1))

        # Inception groups 1
        model.add(Inception([(64,), (96, 128), (16, 32), (32,)]))
        model.add(Inception([(128,), (128, 192), (32, 96), (64,)]))
        model.add(maxPool(3, 3, 2, 2, padding='same', name='pool3'))

        # Inception groups 2
        model.add(Inception([(192,), (96, 208), (16, 48), (64,)]))
        model.add(Inception([(160,), (112, 224), (24, 64), (64,)]))
        model.add(Inception([(128,), (128, 256), (24, 64), (64,)]))
        model.add(Inception([(112,), (144, 288), (32, 64), (64,)]))
        model.add(Inception([(256,), (160, 360), (32, 128), (128,)]))
        model.add(maxPool(3, 3, 2, 2, padding='same'))

        # Inception groups 3
        model.add(Inception([(256,), (160, 320), (32, 128), (128,)]))
        model.add(Inception([(384,), (192, 384), (48, 128), (128,)]))
        model.add(tf.keras.layers.AveragePooling2D(pool_size=(7,7), strides=1, padding='valid'))

        # Output phase
        model.add(flatten())
        model.add(dropOut(self._dropOutProb))
        model.add(fullConnect(None, self._numClass, activation='linear', name='fc1'))
        model.add(fullConnect(None, self._numClass, activation='softmax', name='fc2'))

        return model

    def loadWeights(self, weightsPath):
        self._model.load_weights(weightsPath, by_name=True)

    def train(self, X, Y, validX, validY, 
            weightsSavePath,
            batches = 20000,
            batchSize = 128, 
            learningRate=0.01):

        self._trainSummaryWriter = tf.summary.create_file_writer(weightsSavePath+'train.log')

        tf.keras.backend.set_learning_phase(True)
        self._optimizer = tf.keras.optimizers.RMSprop(learning_rate=learningRate,
                momentum=0.9, epsilon=1.0)
        self._loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
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

            print("Batch: ", i)
            print("TrainResult: ", dict(zip(self._model.metrics_names, trainResult)))

            if i % 20 == 0:
                evaluateResult = self._model.evaluate(validX, validY, verbose=0)
                print("Valid: ", dict(zip(self._model.metrics_names, evaluateResult)))

            if i % 500 == 0:
                filename = weightsSavePath + "googlenet_" + str(i) + '.h5'
                self._model.save_weights(filename)

            if i % 450 == 0:
                self._model.optimizer.lr.assign(self._model.optimizer.learning_rate * 0.94)

            self.writeLogs(i, self._model.metrics_names, trainResult)

        self._model.save_weights(weightsSavePath+"googlenet.h5")

    def writeLogs(self, batch, names, logs):
        with self._trainSummaryWriter.as_default():
            for name, value in zip(names, logs):
                tf.summary.scalar(name, value, step=batch)

    def evaluate(self, X, Y, files):
        tf.keras.backend.set_learning_phase(False)
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
            else:
                print('{} {} {} {}'.format(files[i], predict, maxPred, actual))

        print("accuracy: ", acc/row)
        print("metrics: ", metrics)
            

def conv(kernelHeight, kernelWidth, filters, strideY, strideX, activation='relu', padding='SAME', 
        kernelRegularizer=None, name=None):
    return tf.keras.layers.Conv2D(filters=filters, kernel_size=[kernelHeight, kernelWidth],
            strides=[strideY, strideX], padding=padding, activation=activation, 
            kernel_regularizer=kernelRegularizer, bias_initializer='glorot_uniform', name=name)

def dropOut(dropOutProb):
    return tf.keras.layers.Dropout(dropOutProb)

def flatten():
    return tf.keras.layers.Flatten()

def fullConnect(numIn, numOut, name, activation='relu'):
    return tf.keras.layers.Dense(numOut, input_shape=(numIn,),
            activation=activation, bias_initializer='glorot_uniform', name=name)

def maxPool(kernelHeight, kernelWidth,
        strideY, strideX, padding='SAME', name=None):
    return tf.keras.layers.MaxPool2D(pool_size=(kernelHeight, kernelWidth),
            strides=(strideY, strideX), padding=padding, name=name)