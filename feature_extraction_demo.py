from scipy import misc

from feature_extraction import normalize_text_lines
from text_features import TextSample

input_path = "data/Verdana/45.png"
output_path = "testing.png"
show_line_separations = False
show_connected_components = True

img_array = misc.imread(input_path, flatten=True)

img_array = normalize_text_lines(img_array)
ts = TextSample(img_array)
print(ts.features())

for line_sample in ts.line_samples:
    y = line_sample.coordinates[1]

    if show_line_separations:
        img_array[y + line_sample.scanlines.to, :] = 0
        img_array[y + line_sample.scanlines.ul, :] = 128
        img_array[y + line_sample.scanlines.bl, :] = 128
        img_array[y + line_sample.scanlines.bo, :] = 0

    for cc in line_sample.connected_components:

      if show_connected_components:
        img_array[y + cc.top, cc.coordinates[0]:cc.coordinates[0] + cc.width] = 0
        img_array[y + cc.top:y + cc.bottom, cc.coordinates[0]] = 0
        img_array[y + cc.bottom, cc.coordinates[0]:cc.coordinates[0] + cc.width] = 0
        img_array[y + cc.top:y + cc.bottom, cc.coordinates[0] + cc.width] = 0

misc.imsave(output_path, img_array)
