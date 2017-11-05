import json
from concurrent.futures import ProcessPoolExecutor
from optparse import OptionParser

import matplotlib as mpl
import multiprocessing as mp

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
num_samples = None


def extract_features_from_file(file_path):
    img_data = img_as_float(data.load(file_path))[:, :, 0]
    return compute_feats(img_data, kernels).flatten().tolist()


def extract_features(font_name):
    points = []

    img_files = os.listdir("data/{}".format(font_name))
    i = 0

    if num_samples is not None:
        img_files = img_files[:num_samples]

    progress = [False, False, False]

    for img_file in img_files:
        img_data = img_as_float(data.load(cwd + '/data/{}/{}'.format(font_name, img_file)))[:, :, 0]
        points.append(compute_feats(img_data, kernels).flatten().tolist())

        i += 1

        progress_percent = i / len(img_files)

        if progress_percent > 0.25 and not progress[0]:
            progress[0] = True
            print(font_name + ": 25% done.")
        elif progress_percent > 0.5 and not progress[1]:
            progress[1] = True
            print(font_name + ": 50% done.")
        elif progress_percent > 0.75 and not progress[2]:
            progress[2] = True
            print(font_name + ": 75% done.")

    print(font_name + ": 100% done.")
    return (font_name, points)


if __name__ == '__main__':
    output_file = 'data.json'

    parser = OptionParser()
    parser.add_option("-f", "--font", dest="font",
                      help="generate features for a specific font, instead of all fonts. The data will be appended to "
                           "the output file if it already exists.")
    parser.add_option("-n", "--num-samples", dest="num_samples", type="int",
                      help="only extract features from a certain number of images")
    parser.add_option("-o", dest="output_file",
                      help="alternate output file for data, default is 'data.json'")

    (options, args) = parser.parse_args()

    if options.output_file is not None:
        output_file = options.output_file

    num_samples = options.num_samples

    if options.font is not None:
        print("Processing " + options.font + ".")

        (_, points) = extract_features(options.font)

        if os.path.exists(output_file):
            with open(output_file) as f:
                data_points = json.load(f)
        else:
            data_points = {}

        data_points[options.font] = points

        with open(output_file, 'w') as outfile:
            json.dump(data_points, outfile)

        exit()

    pool = ProcessPoolExecutor()
    results = list(pool.map(extract_features, fonts))

    data_points = {}

    for (font, points) in results:
        data_points[font] = points

    with open(output_file, 'w') as outfile:
        json.dump(data_points, outfile)
