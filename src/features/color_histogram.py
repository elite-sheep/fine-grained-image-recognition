# Copyright 2020 Yuchen Wong

import cv2 as cv
import numpy as np

class ColorHistogram():
    def __init__(self, 
            dividX = 1, 
            dividY = 1, 
            pixelBinNum = 256):
        self._dividX = dividX;
        self._dividY = dividY;
        self._pixelBinNum = pixelBinNum;

    def extract(self, image, normalize = False):
        pixelBinSize = 256 / self._pixelBinNum
        height = image.shape[0]
        width = image.shape[1]
        channel = image.shape[2]

        patchX = int(width / self._dividX)
        patchY = int(height / self._dividY)

        outFeatures = np.zeros(self._dividX * self._dividY * self._pixelBinNum * channel, 
                dtype=np.float64)

        for px in range(self._dividX):
            for py in range(self._dividY):
                for row in range(patchY):
                    for col in range(patchX):
                        curRow = py * patchY + row
                        curCol = px * patchX + col
                        for c in range(channel):
                            pixel = image[curRow, curCol][c]
                            binIndex = pixel / pixelBinSize
                            index = (px * self._dividY + py) * channel * self._pixelBinNum + self._pixelBinNum * c + binIndex
                            outFeatures[int(index)] += 1.0

        if normalize == True:
            outFeatures = outFeatures / np.linalg.norm(outFeatures)

        return outFeatures


