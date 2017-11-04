import json

import matplotlib as mpl
mpl.use('TkAgg')

import numpy as np
from scipy import ndimage as ndi

from skimage import data
from skimage.util import img_as_float
from skimage.filters import gabor_kernel

import os


def compute_feats(image, kernels):
    feats = np.zeros((len(kernels), 2), dtype=np.double)
    for k, kernel in enumerate(kernels):
        filtered = ndi.convolve(image, kernel, mode='wrap')
        feats[k, 0] = filtered.mean()
        feats[k, 1] = filtered.var()
    return feats


# prepare filter bank kernels
kernels = []
for theta in range(4):
    theta = theta / 4. * np.pi
    for sigma in (1, 3):
        for frequency in (0.05, 0.25):
            kernel = np.real(gabor_kernel(frequency, theta=theta,
                                          sigma_x=sigma, sigma_y=sigma))
            kernels.append(kernel)

fonts = os.listdir("data")
cwd = os.getcwd()

data_points = {}

for font_name in fonts:
    data_points[font_name] = []

    img_files = os.listdir("data/{}".format(font_name))
    i = 0

    for img_file in img_files:
        img_data = img_as_float(data.load(cwd + '/data/{}/{}'.format(font_name, img_file)))[:, :, 0]
        data_points[font_name].append(compute_feats(img_data, kernels).flatten().tolist())

        print(font_name + ": " + str(i) + "/{}".format(len(img_files)), end="\r")
        i += 1

    print("                                 ", end="\r")


with open('data.json', 'w') as outfile:
    json.dump(data_points, outfile)