# Copyright 2020 Yuchen Wang

import cv2 as cv
import numpy as np

from features.color_histogram import ColorHistogram

def main():
    colorHist = ColorHistogram(2, 2, 64)

    image = cv.imread('/Users/apple/Desktop/AIMango_sample/sample_image/D-Plant2_0610_3.jpg', 1)

    features = colorHist.extract(image, normalize = True)

    print(features)

main()
