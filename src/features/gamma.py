# Copyright 2020 Yuchen Wong

import cv2 as cv
import numpy as np

class Gamma():
    def __init__(self, gamma = 0.5):
        self._gamma = gamma
        # Use look at table to speed up
        self._lookupTable = np.zeros((1, 256), dtype=np.uint8)

        for i in range(256):
            self._lookupTable[0][i] = np.clip(pow(i / 255.0, gamma) * 255.0, 0, 255)

    def extract(self, image):
        resImage = cv.LUT(image, self._lookupTable)
        return resImage
