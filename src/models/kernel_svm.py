# Copyright 2020 Yuchen Wong

import cv2 as cv
import numpy as np
from sklearn.svm import SVC
from sklearn.grid_search import GridSearchCV

class KernelSVM():
    def __init__(self, C = '1.0',
        gamma = 0.001,
        kernel = 'rbf',
        max_iter = -1,
        grid_search = False):
        self._isTrained = False
        self._C = C
        self._gamma = gamma
        self._kernel = kernel
        self._max_iter = max_iter
        self._grid_search = grid_search
        self._model = SVC(C=self._C,
        gamma=self._gamma,
        kernel=self._kernel,
        max_iter=self._max_iter,
        decision_function_shape='ovr')

    def train(self, X, Y):
        if self._isTrained == True:
            print("SVM:: model have been trained.")
            return

        if self._grid_search == True:
            model = SVC(kernel=self._kernel,max_iter=self._max_iter)
            paramGrid = {'C':[1.0,2.0,3.0,4.0],'gamma':[0.01,0.1,0.3,0.5,0.6]}
            gridSearch = GridSearchCV(model,paramGrid)
            gridSearch.fit(X, Y)
            bestParams = gridSearch.best_estimator_.get_params()
            for para, val in bestParams.items():
                print(para,val)
                self._C = bestParams['C']
                self._gamma = bestParams['gamma']
                self._model = SVC(C=self._C,gamma=self._gamma)

        self._model.fit(X, Y)
        self._isTrained = True

    def valid(self, X, Y, classNum = 3):
        if self._isTrained == False:
            print("SVM::valid(): model have not been trained.")
            return

        prediction = self._model.predict(X)
        dataSize = X.shape[0]
        acc = 0.0
        metrics = np.zeros((classNum, classNum), dtype=np.int16)
        for i in range(dataSize):
            if  prediction[i] == Y[i]:
                acc += 1.0
            metrics[int(Y[i])][int(prediction[i])] += 1

        war = 0.0
        for i in range(classNum):
            war += float(metrics[i][i]) / np.sum(metrics[:,i]) * np.sum(metrics[i,:]) / np.sum(metrics)

        return acc / dataSize, war, metrics

    def test(self, X):
        if self._isTrained == False:
            print("SVM::test(): model have not been trained.")

        return self._model.predict(X)
