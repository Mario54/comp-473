import random

from PIL import Image, ImageDraw, ImageFont
import os


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
]

if not os.path.exists("data"):
    os.makedirs("data")
    os.makedirs("test_data")

num_test_images = 1000

for font_name in fonts:
    if not os.path.exists(os.path.join("data", font_name)):
        os.makedirs(os.path.join("data", font_name))

    i = 0

    for i in range(num_test_images):
        print(font_name + ": " + str(i) + "/{}".format(num_test_images), end="\r")
        generate_img(font_name, "data/{}/{}.png".format(font_name, i))

    print("                                               ", end="\r")
