import random
from optparse import OptionParser

from PIL import Image, ImageDraw, ImageFont
import os

from multiprocessing import Process


def generate_img(font, path):
    REPLACEMENT_CHARACTER = u'\uFFFD'
    NEWLINE_REPLACEMENT_STRING = ' ' + REPLACEMENT_CHARACTER + ' '
    text = ''.join([chr(random.randrange(0, 26) + 97) for _ in range(26)])

    font = ImageFont.truetype(os.path.join("fonts", "{}.ttf".format(font)), 20)
    text = text.replace('\n', NEWLINE_REPLACEMENT_STRING)

    lines = []
    line = u""
    bgcolor = "#FFF"
    color = "#000"
    width = 300

    for word in text.split():
        # print(word)

        if word == REPLACEMENT_CHARACTER:
            lines.append(line[1:])
            line = u""
            lines.append(u"")
        elif font.getsize(line + ' ' + word)[0] <= (width - 3 - 3):
            line += ' ' + word
        else:
            lines.append(line[1:])
            line = u""

            line += ' ' + word

    if len(line) != 0:
        lines.append(line[1:])

    line_height = font.getsize(text)[1]
    img = Image.new("RGBA", (width, 50), bgcolor)
    draw = ImageDraw.Draw(img)
    y = 0

    for line in lines:
        draw.text((3, y), line, color, font=font)
        y += line_height

    img.save(path)


fonts = [
    "Times New Roman",
    "Arial",
    "Comic Sans MS",
    "Courier New",
    "Trebuchet MS",
    "Verdana",
    "EBGaramond"
]

if not os.path.exists("data"):
    os.makedirs("data")

num_test_images = 1000


def generate_images(font_name):
    if not os.path.exists(os.path.join("data", font_name)):
        os.makedirs(os.path.join("data", font_name))

    fifty_percent = False

    for i in range(num_test_images):
        # print(font_name + ": " + str(i) + "/{}".format(num_test_images), end="\r")
        generate_img(font_name, "data/{}/{}.png".format(font_name, i))

        if i / num_test_images > 0.5 and not fifty_percent:
            fifty_percent = True
            print(font_name + ": 50% done.")
        elif (i + 1) == num_test_images:
            print(font_name + ": 100% done.")

            # print("                                               ", end="\r")


if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option("-f", "--font", dest="font",
                      help="generate images for a specific font, instead of all fonts")
    parser.add_option("-n", dest="num_images", type="int",
                      help="number of images to generate for each font. Default is 1000")

    (options, args) = parser.parse_args()

    if options.num_images is not None:
        num_test_images = options.num_images

    if options.font is not None:
        generate_images(options.font)
        exit()

    processes = []

    for font_name in fonts:
        # generate_images(font_name)
        p = Process(target=generate_images, args=(font_name,))
        p.start()
        processes.append(p)

    for proc in processes:
        proc.join()
