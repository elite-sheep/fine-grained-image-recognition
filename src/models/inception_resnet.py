# Copyright 2020 Yuchen Wong

import numpy as np
import tensorflow as tf

class InceptionResnet(object):
    def __init__(self, 
            inputShape=[299, 299, 3],
            numClass=3,
            dropOutProb=0.2,
            pretrainedWeights=None):
        self._inputShape = inputShape
        self._numClass = numClass
        self._dropOutProb = dropOutProb
        self._pretrainedWeight = pretrainedWeights

        self._model = self.buildUpModel()

    def buildUpModel(self):
        model = tf.keras.applications.InceptionResNetV2(
            input_shape=self._inputShape,
            weights = None,
            classes = self._numClass)

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

            if i % 500 == 0:
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
