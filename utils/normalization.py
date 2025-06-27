"""
Normalization is used to standardize the intensity values in an image by adjusting the range of
gray level values so that they extend in a desired range of values and improve the contrast of the
image. The main goal of normalization is to reduce the variance of the gray level value along the
ridges to facilitate subsequent processing steps
"""
from math import sqrt
import numpy as np


def normalize_pixel(x, v0, v, m, m0):
    dev_coeff = sqrt((v0 * ((x - m)**2)) / v)
    return m0 + dev_coeff if x > m else m0 - dev_coeff

def normalize(im, m0, v0):
    m = np.mean(im)
    v = np.std(im) ** 2
    (y, x) = im.shape
    normilize_image = im.copy()
    for i in range(x):
        for j in range(y):
            normilize_image[j, i] = normalize_pixel(im[j, i], v0, v, m, m0)

    return normilize_image
