import itertools
import os
import random
from optparse import OptionParser
from os.path import basename

from PIL import Image, ImageDraw, ImageFont
from random_words import LoremIpsum

import helpers

characters = []

i = 0

for c in range(ord('a'), ord('z') + 1):
    characters.append(chr(c))

for c in range(ord('A'), ord('Z') + 1):
    characters.append(chr(c))

characters.append(' ')
characters.append(' ')
characters.append(' ')
characters.append(' ')
characters.append(' ')
characters.append(' ')
characters.append(' ')

li = LoremIpsum()

REPLACEMENT_CHARACTER = u'\uFFFD'
NEWLINE_REPLACEMENT_STRING = ' ' + REPLACEMENT_CHARACTER + ' '


# text = """
# Neil Gaiman's family is of Polish-Jewish and other Eastern European-Jewish origins; his great-grandfather emigrated from Antwerp, Belgium, to the UK before and his grandfather eventually settled in the south of England in the Hampshire city of Portsmouth and established a chain of grocery stores. His father, David Bernard Gaiman, worked in the same chain of stores; his mother, Sheila Gaiman (née Goldman), was a pharmacist. He has two younger sisters, Claire and Lizzy. After living for a period in the nearby town of Portchester, Hampshire, where Neil was born in, the Gaimans moved in to the West Sussex town of East Grinstead where his parents studied Dianetics at the Scientology centre in the town; one of Gaiman's sisters works for the Church of Scientology in Los Angeles. His other sister, Lizzy Calcioli, has said, "Most of our social activities were involved with Scientology or our Jewish family. It would get very confusing when people would ask my religion as a kid. I’d say, 'I’m a Jewish Scientologist.'" Gaiman says that he is not a Scientologist, and that like Judaism, Scientology is his family's religion. About his personal views, Gaiman has stated, "I think we can say that God exists in the DC Universe. I would not stand up and beat the drum for the existence of God in this universe. I don't know, I think there's probably a chance. It doesn't really matter to me."
# """

def generate_img(font, path, ext="ttf"):
    text = li.get_sentences(10)

    text_size = random.choice([25, 30, 35])

    font_path = os.path.join("fonts", "{}.{}".format(font, ext))
    print(font_path)
    font = ImageFont.truetype(font_path, text_size)
    text = text.replace('\n', NEWLINE_REPLACEMENT_STRING)

    lines = []
    line = u""
    bgcolor = "#FFF"
    color = "#000"
    size = (int(700 * text_size/24), int(700 * text_size/24))

    for word in text.split():
        if word == REPLACEMENT_CHARACTER:
            lines.append(line[1:])
            line = u""
            lines.append(u"")
        elif font.getsize(line + ' ' + word)[0] <= (size[0] - 3 - 3):
            line += ' ' + word
        else:
            lines.append(line[1:])
            line = u""

            line += ' ' + word

    if len(line) != 0:
        lines.append(line[1:])

    line_height = font.getsize(text)[1] + 5*(text_size/24)
    img = Image.new("RGBA", (size[0], size[1]), bgcolor)
    draw = ImageDraw.Draw(img)
    y = 0

    for line in lines:
        draw.text((3, y), line, color, font=font)
        y += line_height

    img.save(path)


if not os.path.exists("data"):
    os.makedirs("data")

num_test_images = 1000


def generate_image(font_name, index, ext="ttf"):
    generate_img(font_name, "data/{}/{}.png".format(font_name, index), ext=ext)


def generate_font_image(font_name, output=None):
    if output is None:
        output = "test.png"

    generate_img(font_name, output)


if __name__ == '__main__':
    fonts = [(os.path.splitext(basename(f))[0], os.path.splitext(basename(f))[1]) for f in os.listdir("fonts")]

    parser = OptionParser()
    parser.add_option("-f", "--font", dest="font",
                      help="generate images for a specific font, instead of all fonts")
    parser.add_option("-n", dest="num_images", type="int",
                      help="number of images to generate for each font. Default is 1000")
    parser.add_option("-o", dest="output_path")

    (options, args) = parser.parse_args()

    if options.num_images is not None:
        num_test_images = options.num_images

    if options.font is not None:
        if num_test_images == 1:
            generate_font_image(options.font, output=options.output_path)
            exit()
        else:
            fonts = [options.font]

    for (font_name, ext) in fonts:
        if not os.path.exists(os.path.join("data", font_name)):
            os.makedirs(os.path.join("data", font_name))

    parallel_data = [zip(range(num_test_images), itertools.repeat(font_name)) for font_name in fonts]
    flat_parallel_data = [item for sublist in parallel_data for item in sublist]

    generate_image_args = [{"font_name": font_name, "index": i, "ext": ext} for (i, (font_name, ext)) in flat_parallel_data]

    helpers.parallel_process(generate_image_args, generate_image, use_kwargs=True)
