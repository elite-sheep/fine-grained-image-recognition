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
            activation='relu', padding='valid'))
        model.add(tf.keras.layers.BatchNormalization(momentum=0.9))
        model.add(maxPool(3, 3, 2, 2, padding='valid', name='pool1'))

        # 2nd layer
        model.add(conv(5, 5, 256, 1, 1, name='conv2'))
        model.add(tf.keras.layers.BatchNormalization(momentum=0.9))
        model.add(maxPool(3, 3, 2, 2, padding='valid', name='pool2'))

        # 3rd layer
        model.add(conv(3, 3, 384, 1, 1, name='conv3'))

        # 4th layer
        model.add(conv(3, 3, 256, 1, 1, name='conv4'))

        # 5th layer
        model.add(conv(3, 3, 256, 1, 1, name='conv5'))
        model.add(maxPool(3, 3, 2, 2, padding='valid', name='pool5'))

        # 6th layer
        model.add(flatten())
        model.add(fullConnect(6*6*256, 4096, name='fc6'))
        model.add(dropOut(1.0 - self._dropOutProb))

        # 7th layer
        model.add(fullConnect(4096, 4096, name='fc7'))
        model.add(dropOut(1.0 - self._dropOutProb))

        # 8th layer
        model.add(fullConnect(4096, self._numClass, activation='softmax', name='fc8'))

        return model

    def train(self, X, Y, validX, validY, 
            weightsSavePath,
            batches = 20000,
            batchSize = 128, 
            learningRate=0.01, 
            decayStep=[1000]):

        tf.keras.backend.set_learning_phase(True)
        self._optimizer = tf.keras.optimizers.SGD(learning_rate=learningRate,
                momentum=0.9, nesterov=True)
        self._loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
        self._model.compile(optimizer=self._optimizer, loss=self._loss,
                metrics=[tf.keras.metrics.categorical_accuracy])
        self._model.build()
        self._model.summary()

        curDecayStep = 0

#        self._model.fit(X, Y, batch_size=batchSize, epochs=int(batches*128/5600), 
#                validation_data=(validX, validY))

        for i in range(batches):
            batchIndicies = np.random.choice(X.shape[0], batchSize)
            curX = X[batchIndicies].copy()
            curY = Y[batchIndicies].copy()
            trainResult = self._model.train_on_batch(curX, curY)

            print("Batch: ", i)
            print("TrainResult: ", dict(zip(self._model.metrics_names, trainResult)))

            if i % 10 == 0:
                evaluateResult = self._model.evaluate(validX, validY)
                print("Valid: ", dict(zip(self._model.metrics_names, evaluateResult)))

            if i % 500 == 0:
                filename = weightsSavePath + "alexnet_" + str(i)
                self._model.save_weights(filename)

            if curDecayStep < len(decayStep) and i == decayStep[curDecayStep]:
                self._model.optimizer.lr.assign(self._model.optimizer.learning_rate * 0.5)
                curDecayStep += 1

    def evaluate(self, X, Y):
        tf.keras.backend.set_learning_phase(False)
        row = X.shape[0]

        predictions = self._model.predict(X)
        acc = 0.0
        metrics = np.zeros((self._numClass, self._numClass), np.int32)
        for i in range(row):
            predict = predictions[i]
            maxPred = tf.math.argmax(predict)
            actual = tf.math.argmax(Y[i])
            
            print(predict)

            metrics[int(actual)][int(maxPred)] += 1
            if int(maxPred) == int(actual):
                acc += 1.0

        print("accuracy: ", acc/row)
        print("metrics: ", metrics)
            

def conv(kernelHeight, kernelWidth, filters, strideY, strideX, padding='SAME', name=None):
    return tf.keras.layers.Conv2D(filters=filters, kernel_size=[kernelHeight, kernelWidth],
            strides=[strideY, strideX], padding=padding, activation='relu', name=name)

def dropOut(keepProb):
    return tf.keras.layers.Dropout(keepProb)

def flatten():
    return tf.keras.layers.Flatten()

def fullConnect(numIn, numOut, name, activation='relu'):
    return tf.keras.layers.Dense(numOut, input_shape=(numIn,),
            activation=activation, name=name)

def maxPool(kernelHeight, kernelWidth,
        strideY, strideX, padding='SAME', name=None):
    return tf.keras.layers.MaxPool2D(pool_size=(kernelHeight, kernelWidth),
            strides=(strideY, strideX), padding=padding, name=name)
