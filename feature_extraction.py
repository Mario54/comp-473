import itertools
import json
from optparse import OptionParser

import matplotlib as mpl
from scipy import misc

import helpers
import preprocessing
from text_features import TextSample, average

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


def get_filter_kernels():
    """
    Returns the Gabor filters used to compute the features of an image.
    """
    # prepare filter bank kernels
    kernels = []
    for theta in range(4):
        theta = theta / 4. * np.pi
        for sigma in (1,):
            for frequency in (0.05, 0.25):
                kernel = np.real(gabor_kernel(frequency, theta=theta,
                                              sigma_x=sigma, sigma_y=sigma))
                kernels.append(kernel)

    return kernels


kernels = get_filter_kernels()

cwd = os.getcwd()
num_samples = None
algo = 'gabor'

def normalize_text_lines(raw_data):
    breakpoints = preprocessing._partition_with_projection_profile(raw_data)

    line_height = average([(end - start) for (start, end) in breakpoints], default=1)

    for (start, end) in breakpoints:
        try:
            raw_data[start - 1, :] = 255
            raw_data[end + 1, :] = 255
        except IndexError:
            continue

    img_array = misc.imresize(raw_data, 20 / line_height)

    return img_array

def extract_features_from_file(file_path, algo='gabor'):
    raw_data = misc.imread(file_path, flatten=True)

    if algo == 'text':
        return TextSample(normalize_text_lines(raw_data)).features()

    img_data = preprocessing.normalize_sample(raw_data)
    return compute_feats(img_data, kernels).flatten().tolist()


def generate_data(f, i, algo):
    return f, extract_features_from_file(cwd + '/data/{}/{}.png'.format(f, i), algo)


def generate_all_font_data(font_name):
    points = []

    img_files = os.listdir("data/{}".format(font_name))

    if num_samples is not None:
        img_files = img_files[:num_samples]

    for img_file in img_files:
        img_data = img_as_float(data.load(cwd + '/data/{}/{}'.format(font_name, img_file)))[:, :, 0]
        points.append(compute_feats(img_data, kernels).flatten().tolist())

    return font_name, points


if __name__ == '__main__':
    fonts = os.listdir("data")

    output_file = 'data.json'

    parser = OptionParser()
    parser.add_option("-f", "--font", dest="font",
                      help="generate features for a specific font, instead of all fonts. The data will be appended to "
                           "the output file if it already exists.")
    parser.add_option("-n", "--num-samples", dest="num_samples", type="int",
                      help="only extract features from a certain number of images")
    parser.add_option("--file", dest="file",
                      help="Print features for a file.")
    parser.add_option("-o", dest="output_file",
                      help="alternate output file for data, default is 'data.json'")
    parser.add_option("--algo", dest="algo",
                      help="If 'text' is used, features will be extracted from handpicked font features")

    (options, args) = parser.parse_args()

    if options.algo is not None:
        algo = options.algo

    if options.file is not None:
        print(extract_features_from_file(options.file, algo=algo))
        exit()

    if options.output_file is not None:
        output_file = options.output_file

    num_samples = options.num_samples if options.num_samples is not None else 500

    if options.font is not None:
        print("Processing " + options.font + ".")

        extract_features_args = [{"f": font_name, "i": i, "algo": algo} for (i, font_name) in
                                 list(zip(range(num_samples), itertools.repeat(options.font)))]

        results = helpers.parallel_process(extract_features_args, generate_data, use_kwargs=True)

        points = [point for (_, point) in results]

        if os.path.exists(output_file):
            with open(output_file) as f:
                data_points = json.load(f)
        else:
            data_points = {}

        data_points[options.font] = points

        with open(output_file, 'w') as outfile:
            json.dump(data_points, outfile)

        exit()

    print("Extracting {} features.".format(algo))

    files = [zip(range(num_samples), itertools.repeat(font_name)) for font_name in fonts]
    flat_list = [item for sublist in files for item in sublist]

    extract_features_args = [{"f": font_name, "i": i, "algo": algo} for (i, font_name) in flat_list]

    results = helpers.parallel_process(extract_features_args, generate_data, use_kwargs=True, n_jobs=1)

    data_points = {font_name: [] for font_name in fonts}

    for r in results:
        try:
            (font, point) = r
            data_points[font].append(point)
        except TypeError:
            print(r)

    with open(output_file, 'w') as outfile:
        json.dump(data_points, outfile)
