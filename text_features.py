import collections
import math

import numpy as np

from preprocessing import _partition_with_projection_profile

ScanLines = collections.namedtuple('ScanLines', ['bo', 'bl', 'ul', 'to'])

PIXEL_THRESHOLD = 200

def average(lst, default=0):
    if len(lst) == 0:
        return default

    return sum(lst) / len(lst)


class TextSample:
    def __init__(self, matrix):
        self.line_samples = []  # Type: list[LineSample]

        for (start, end) in _partition_with_projection_profile(matrix):
            ls = LineSample(
                matrix[start:end],
                coordinates=(0, start),
                # color_mean=matrix.mean(),
            )

            if ls.normalized_height > 0 and len(ls.connected_components) > 0:
                self.line_samples.append(ls)

    def projection_profile_derivative(self):
        return average([l.projection_profile_derivative() for l in self.line_samples])

    def projection_profile_density(self):
        return average([l.projection_profile_density() for l in self.line_samples])

    def horizontal_projection_profile_density(self):
        return sum([line.horizontal_projection_profile_density() for line in self.line_samples]) / len(
            self.line_samples)

    def horizontal_projection_profile_derivative(self):
        return sum([line.horizontal_projection_profile_derivative() for line in self.line_samples]) / len(
            self.line_samples)

    def avg_normalized_heights_connected_components(self):
        return average([l.avg_normalized_heights_connected_components() for l in self.line_samples])

    def avg_width_connected_components(self):
        return average([cc.width for cc in self.connected_components()])

    def avg_space_between_connected_components(self):
        return average([l.avg_space_between_connected_components() for l in self.line_samples])

    def avg_width_horizontal_black_runs(self):
        return average([l.avg_width_horizontal_black_runs() for l in self.line_samples])

    def avg_width_vertical_black_runs(self):
        return average([l.avg_width_vertical_black_runs() for l in self.line_samples])

    def connected_components(self):
        ccs = [l.connected_components for l in self.line_samples]
        return [item for sublist in ccs for item in sublist]

    def density_diff_between_consecutive_scan_lines(self):
        return average([cc.scanline_difference_density() for cc in self.connected_components()])

    def avg_scanline_height(self):
        return average([l.middle_scanline_size for l in self.line_samples])

    def features(self):
        return [
            self.projection_profile_density(),
            self.projection_profile_derivative(),
            self.avg_normalized_heights_connected_components(),
            self.avg_width_connected_components(),
            self.avg_space_between_connected_components(),
            # self.avg_width_vertical_black_runs(),
            # self.avg_width_horizontal_black_runs(),
            self.density_diff_between_consecutive_scan_lines(),
            self.avg_scanline_height()
        ]


class LineSample:
    def __init__(self, line_matrix, coordinates):
        self.connected_components = []  # Type: list[ConnectedComponent]
        self.coordinates = coordinates
        self.data = line_matrix

        vfunc = np.vectorize(lambda d: 0 if d >= PIXEL_THRESHOLD else 1)
        vectorized_data = vfunc(line_matrix)

        vertical_profile = vectorized_data.sum(axis=1)

        ul = self._get_max_vertical_profile(vertical_profile, range(0, int(math.floor(len(line_matrix) / 2))))
        bl = self._get_max_vertical_profile(vertical_profile,
                                            range(int(math.floor(len(line_matrix) / 2)), len(line_matrix)))

        self.normalized_height = abs(ul - bl)

        self.scanlines = ScanLines(bo=len(line_matrix), to=0, ul=ul, bl=bl)

        self.middle_scanline_size = self.scanlines.bl - self.scanlines.ul

        for (start, end) in _partition_with_projection_profile(line_matrix, axis=0, threshold=0.95):
            cc = ConnectedComponent(data=line_matrix[:, start:end], scanlines=self.scanlines,
                                    coordinates=(start, coordinates[1]))

            if cc.width > 1 and cc.height() > 1:
                self.connected_components.append(cc)

    def _get_max_vertical_profile(self, vertical_profile, index_range):
        max_i = 0
        min_vertical_profile = -math.inf

        for i in index_range:
            try:
                current_profile = abs(vertical_profile[i + 1] - vertical_profile[i])
                if current_profile > min_vertical_profile:
                    max_i = i
                    min_vertical_profile = current_profile
            except IndexError:
                break

        return max_i + 1

    def projection_profile(self):
        data_copy = np.array(self.data, copy=True)
        # vfunc = np.vectorize(lambda d: 0 if d >= PIXEL_THRESHOLD else 1)
        vfunc = np.vectorize(lambda d : 255 - d)
        data_copy = vfunc(data_copy)

        horizontal_profile =  data_copy[self.scanlines.ul:self.scanlines.bl + 1].sum(axis=0)

        start = 0
        end = len(horizontal_profile) - 1

        while horizontal_profile[start] == 0:
            start += 1

        while horizontal_profile[end] == 0:
            end -= 1

        # color_mean = self.data[self.scanlines.ul:self.scanlines.bl + 1].mean()

        return horizontal_profile[start:end+1], len(horizontal_profile[start:end+1])

    def projection_profile_density(self):
        pp, length = self.projection_profile()

        return sum(pp) / length

    def projection_profile_derivative(self):
        pp, length = self.projection_profile()

        s = 0

        for i in range(len(pp) - 1):
            s += (pp[i + 1] - pp[i]) ** 2

        return s / length

    def avg_normalized_heights_connected_components(self):
        result = 0
        n = 0

        (bo, bl, ul, to) = self.scanlines

        for cc in self.connected_components:
            if cc.typographical_class == "Full":
                n += 1
                result += cc.height() * abs(bl - ul) / abs(bo - to)
            elif cc.typographical_class == "High":
                n += 1
                result += cc.height() * abs(bl - ul) / abs(bl - to)
            elif cc.typographical_class == "Short":
                n += 1
                result += cc.height()
            elif cc.typographical_class == "Deep":
                n += 1
                result += cc.height() * abs(bl - ul) / abs(bo - ul)

        if n == 0:
            return sum([cc.height() for cc in self.connected_components]) / len(self.connected_components)

        return result / n

    def avg_width_connected_components(self):
        result = 0
        n = 0

        for cc in self.connected_components:
            if cc.typographical_class == "Short" and cc.morphological_class == "Square":
                n += 1
                result += cc.width

        if n == 0:
            return sum([cc.width for cc in self.connected_components]) / len(self.connected_components)

        return result / n

    def avg_space_between_connected_components(self):
        result = 0
        n = 0

        for i in range(len(self.connected_components) - 1):
            cc1 = self.connected_components[i]
            cc2 = self.connected_components[i + 1]

            space_between = cc2.coordinates[0] - (cc1.coordinates[0] + cc1.width)

            if space_between < abs(self.scanlines.bl - self.scanlines.ul):
                n += 1
                result += space_between

        if n == 0:
            return 0

        return result / n

    def avg_width_horizontal_black_runs(self):
        result = 0
        n = 0

        for cc in self.connected_components:
            for horizontal_black_run_width in cc.horizontal_black_runs():
                n += 1
                result += horizontal_black_run_width

        if n == 0:
            return 0

        return result / n

    def avg_width_vertical_black_runs(self):
        result = 0
        n = 0

        for cc in self.connected_components:
            for horizontal_black_run_width in cc.vertical_black_runs():
                n += 1
                result += horizontal_black_run_width

        if n == 0:
            return 0

        return result / n

    def scanline_difference_density(self):
        if len(self.connected_components) == 0:
            return 0

        return sum(cc.scanline_difference_density() for cc in self.connected_components if cc.typographical_class != "Sub" and cc.typographical_class != "Sup") / len(
            self.connected_components)


class ConnectedComponent:
    def __init__(self, data, scanlines, coordinates):
        (_, width) = data.shape
        self.coordinates = coordinates
        self.width = width
        self.scanlines = scanlines
        self.data = data

        vertical_profile = data.sum(axis=1)
        self.top = 0
        self.bottom = len(vertical_profile)

        for i, scanline_sum in enumerate(vertical_profile):
            if scanline_sum < 255 * width * 0.999:
                self.top = i
                break

        for i in reversed(range(0, len(vertical_profile))):
            scanline_sum = vertical_profile[i]
            if scanline_sum < 255 * width * 0.999:
                self.bottom = i
                break

        self.epsilon = 1 / 12 * abs(self.top - self.bottom)

        self.typographical_class = self._calculate_typographical_class()
        self.morphological_class = self._calculate_morpholigcal_class()

    def _calculate_typographical_class(self):
        if abs(self.top - self.scanlines.to) <= self.epsilon and abs(self.bottom - self.scanlines.bo) <= self.epsilon:
            return 'Full'
        elif abs(self.top - self.scanlines.to) <= self.epsilon and abs(self.bottom - self.scanlines.bl) <= self.epsilon:
            return 'High'
        elif abs(self.top - self.scanlines.ul) <= self.epsilon and abs(self.scanlines.bo - self.bottom) <= self.epsilon:
            return 'Deep'
        elif abs(self.top - self.scanlines.ul) <= self.epsilon and abs(self.bottom - self.scanlines.bl) <= self.epsilon:
            return 'Short'
        elif abs(self.top - self.scanlines.to) <= self.epsilon and abs(self.scanlines.ul - self.bottom) <= self.epsilon:
            return 'Sup'
        elif abs(self.top - self.scanlines.bl) <= self.epsilon and abs(self.scanlines.bo - self.bottom) <= self.epsilon:
            return 'Sub'
        else:
            return 'N/A'

    def _calculate_morpholigcal_class(self):
        if self.height() <= (1 / 25) * abs(self.scanlines.bo - self.scanlines.to):
            return 'Small'

        r = self.width / self.height()

        if r >= 1.75:
            return 'Wide'
        elif r > 1.25:
            return 'Large'
        elif r > 0.75:
            return 'Square'
        elif r > 0.5:
            return 'Tall'
        elif r < 0.5:
            return 'Thin'

    def height(self):
        return abs(self.top - self.bottom)

    def horizontal_black_runs(self):
        vfunc = np.vectorize(lambda d: 0 if d >= 128 else 1)
        return self._get_black_runs(vfunc(self.data[math.floor((1 / 2) * abs(self.top - self.bottom)), :]),
                                    size_range=((1 / 8) * self.width, (1 / 2) * self.width))

    def _get_black_runs(self, array, size_range):
        black_runs = []
        current_black_run_start = -1

        for i, e in enumerate(array):
            if e == 1:
                if current_black_run_start == -1:
                    current_black_run_start = i
            else:
                if current_black_run_start != -1:
                    black_run_width = i - current_black_run_start

                    if size_range[0] <= black_run_width <= size_range[1]:
                        black_runs.append(black_run_width)

                    current_black_run_start = -1

        if current_black_run_start != -1:
            black_run_width = len(array) - current_black_run_start
            if size_range[0] <= black_run_width <= size_range[1]:
                black_runs.append(black_run_width)

        return black_runs

    def vertical_black_runs(self):
        vfunc = np.vectorize(lambda d: 0 if d >= 128 else 1)

        return self._get_black_runs(vfunc(self.data[self.top:self.bottom, math.floor((1 / 2) * self.width)]),
                                    size_range=((1 / 8) * self.height(), (1 / 3) * self.height()))

    def _scanline_difference(self, matrix):
        return np.absolute(np.matrix(
            [np.subtract(matrix[i], matrix[i + 1]) for i in range(matrix.shape[0] - 1)]
        ))

    def scanline_difference_density(self):
        d = self.data

        regions = [
            d[self.top:self.top + int((1 / 3) * self.height()) + 1],
            d[self.bottom - int((1 / 3) * self.height()) - 1:self.bottom]
        ]

        color_mean = 255 - self.data.mean()

        num_pixels = 0

        for r in regions:
            num_pixels += abs(self._scanline_difference(r).sum())

        if abs(color_mean) < 0.5:
            print("hi!")

        return num_pixels / self.width / color_mean
