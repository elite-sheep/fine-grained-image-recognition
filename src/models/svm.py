# Copyright 2020 Yuchen Wong

import cv2 as cv
import numpy as np
from sklearn.svm import LinearSVC

class SVM():
    def __init__(self, penalty = 'l2',
            loss = 'squared_hinge',
            C = '1.0',
            multiClass = 'ovr',
            maxIter = 1000):
        self._isTrained = False
        self._penalty = penalty
        self._loss = loss
        self._C = C
        self._multiClass = multiClass
        self._maxIter = maxIter
        self._model = LinearSVC(penalty=self._penalty,
                loss=self._loss,
                C=self._C,
                multi_class=self._multiClass,
                max_iter=self._maxIter,
                verbose=1)

    def train(self, X, Y):
        if self._isTrained == True:
            print("SVM:: model have been trained.")
            return

        self._model.fit(X, Y)
        self._isTrained = True

    def valid(self, X, Y):
        if self._isTrained == False:
            print("SVM::valid(): model have not been trained.")
            return

        return self._model.score(X, Y)

    def test(self, X):
        if self._isTrained == False:
            print("SVM::test(): model have not been trained.")

        return self._model.predict(X)
