from optparse import OptionParser

import numpy as np
from scipy import misc
import itertools
import math
import os
import random

import helpers

wanted_line_height = 16
wanted_width = 256
wanted_height = 256


def derivative(d):
    l = []

    for i in range(len(d) - 1):
        l.append(d[i + 1] - d[i])

    return l


PIXEL_THRESHOLD = 128


def _partition_with_projection_profile(img_array, threshold=0.999, axis=1):
    """

    :param img_array:
    :param threshold:
    :return:
    """
    vfunc = np.vectorize(lambda d: 0 if d >= PIXEL_THRESHOLD else 1)

    raw_data = vfunc(img_array)
    # (height, width) = raw_data.shape

    row_sums = raw_data.sum(axis=axis)

    # threshold = (width if axis == 1 else height) * 255 * threshold

    row_coordinates = []
    current_row_start = -1

    for i, sum in enumerate(row_sums):

        if sum > 0:
            if current_row_start == -1:
                current_row_start = i
                continue
        else:
            if current_row_start >= 0:
                row_coordinates.append((current_row_start, i))
                current_row_start = -1

    return row_coordinates


def _fill_line(ranges, max_width):
    whitespace = remaining_whitespace(ranges, max_width)
    infinite_ranges = itertools.cycle(ranges)

    while True:
        (start, end) = next(infinite_ranges)

        if (end - start) <= whitespace:
            ranges.append((start, end))
            whitespace -= (end - start)
        else:
            break

    if whitespace > 0:
        for (start, end) in ranges:
            if (end - start) <= whitespace:
                ranges.append((start, end))
                whitespace -= (end - start)

    return ranges


def remaining_whitespace(ranges, max_width):
    total_width = sum([end - start for (start, end) in ranges])
    return max_width - total_width - 1


def slice_ranges(ranges, max_width):
    lines = []
    current_line_ranges = []

    for (start, end) in ranges:
        if (end - start) > remaining_whitespace(current_line_ranges, max_width):
            lines.append(current_line_ranges)
            current_line_ranges = [(start, end)]
        else:
            current_line_ranges.append((start, end))

    if len(current_line_ranges) > 2:
        lines.append(current_line_ranges)

    for (i, l) in enumerate(lines):
        lines[i] = _fill_line(l, max_width)

    return lines


def _normalize_text_line(line_array, line_height=None):
    (height, _) = line_array.shape
    line_height = line_height if line_height is not None else height

    normalized_line = np.pad(line_array,
                             (math.ceil((line_height - height) / 2), math.floor((line_height - height) / 2)),
                             'constant', constant_values=255)
    normalized_line = misc.imresize(normalized_line, wanted_line_height / line_height)

    line_transpose = np.transpose(normalized_line)

    character_ranges = _partition_with_projection_profile(line_transpose, threshold=0.94)

    broken_up_ranges = slice_ranges(character_ranges, max_width=wanted_height)

    lines = []

    for ranges in broken_up_ranges:
        l = np.transpose(_shrink(line_transpose, ranges))
        l = np.pad(l, [(0, 0), (0, remaining_whitespace(ranges, wanted_width))], 'constant', constant_values=255)
        lines.append(l)

    return lines


def _shrink(array, ranges_to_keep):
    (_, width) = array.shape

    result = np.matrix(
        [[255] * width]
    )

    for (start, end) in ranges_to_keep:
        result = np.vstack((result, array[start:end]))

    return result


def normalize_sample(img_array):
    result = None

    line_coordinates = _partition_with_projection_profile(img_array)

    line_height = max([end - start for (start, end) in line_coordinates])
    line_bag = []

    for (start, end) in line_coordinates:
        for line in _normalize_text_line(img_array[start:end], line_height=line_height):
            line_bag.append(line)

            if result is not None:
                result = np.vstack((result, line))
            else:
                result = np.matrix(line)

            if result is not None and result.shape[0] >= wanted_height:
                break

        if result is not None and result.shape[0] >= wanted_height:
            break

    while result.shape[0] < wanted_height:
        result = np.vstack((result, random.choice(line_bag)))

    return result


def preprocess_file(input_path, output_path):
    img_array = misc.imread(input_path, flatten=True)

    normalized = normalize_sample(img_array)

    misc.imsave(output_path, normalized)


if __name__ == "__main__":
    cwd = os.getcwd()

    parser = OptionParser()
    parser.add_option("--file", dest="file",
                      help="preprocess one file.")
    parser.add_option("--folder", dest="folder", help="Preprocess all files in the folder")

    parser.add_option("-o", dest="output_path")

    (options, args) = parser.parse_args()

    if options.file is not None:
        if options.output_path is None:
            print("-o option is required when preprocessing one file")
            exit()

        preprocess_file(options.file, options.output_path)
        exit()
    else:
        folder = options.folder if options.folder is not None else "data"
        fonts = os.listdir(folder)

        preprocessing_args = []

        if not os.path.exists("data-preprocessed"):
            os.makedirs("data-preprocessed")

        for font_name in fonts:
            path = cwd + "/" + folder + "/" + font_name
            output = cwd + "/" + "data-preprocessed" + "/" + font_name
            preprocessing_args = preprocessing_args + [{"input_path": path + "/" + f, "output_path": output + "/" + f}
                                                       for f in os.listdir(path)]

            if not os.path.exists(output):
                os.makedirs(output)

        helpers.parallel_process(preprocessing_args, preprocess_file, use_kwargs=True, n_jobs=1)
