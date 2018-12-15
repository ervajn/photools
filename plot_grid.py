#!/usr/bin/env python
from __future__ import division, print_function
import argparse
import logging
import math
import os
import PIL
from PIL import ImageFont
from PIL import ImageDraw
import imageutils

def get_font(font_height):
    fonts = ["Ubuntu-B.ttf", "OpenSans-Regular.ttf", "arial.ttf"]
    for f in fonts:
        try:
            return PIL.ImageFont.truetype(f, font_height)
        except OSError:
            pass
    logging.error('No TrueType font available. Tried {}. Using default font'.format(fonts))
    return PIL.ImageFont.load_default()


def plot_grid(images, texts=None, tile_size=120):
    n = int(math.ceil(math.sqrt(len(images))))
    full_size = tile_size * n
    grid_image = PIL.Image.new('RGB', (full_size, full_size))
    for i, img in enumerate(images):
        logging.debug('Adding image {}({}) {}'.format(i, len(images), img))
        x, y = (i % n) * tile_size, (i // n) * tile_size
        tile = imageutils.load_image(img)
        margin = abs((tile.width - tile.height) // 2)
        if (tile.width > tile.height):
            tile = tile.crop((margin, 0, margin + tile.height, tile.height))
        else:
            tile = tile.crop((0, margin, tile.width, margin + tile.width))
        tile = tile.resize((tile_size, tile_size), PIL.Image.ANTIALIAS)
        draw = PIL.ImageDraw.Draw(tile)
        if texts:
            for font_height in range(tile_size // 5, 1, -1):
                font = get_font(font_height)
                text_width, _ = font.getsize(texts[i])
                if text_width < tile_size:
                    break
            draw.text((0, tile_size - font_height), texts[i], (255, 255, 0), font=font)
        grid_image.paste(tile, (x, y))

    return grid_image

def parse_arguments():
    parser = argparse.ArgumentParser(description='Plot images as a grid',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-v', '--verbose', action='count')
    parser.add_argument('-d', '--directory', default='.',
                        help='The directory with pictures')
    parser.add_argument('-f', '--file', default='grid.jpg',
                        help='Resulting grid image file name')
    parser.add_argument('-s', '--size', type=int, default=160,
                        help='Size of each tile')
    parser.add_argument('-t', '--text', action='store_true', default=False,
                        help='Print information text on each image')

    args = parser.parse_args()
    return args

def _main():
    args = parse_arguments()
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=level)

    images = [os.path.join(dp, f) for dp, dn, filenames in
              os.walk(args.directory) for f in filenames if os.path.splitext(f)[1].lower() in ['.jpg','.png','.jpeg']]
    logging.debug('Found {} images in {}'.format(len(images), args.directory))
    if images == []:
        logging.error('No images found in {}'.format(args.directory))
        exit(-1)

    texts=[os.path.basename(f) for f in images] if args.text else None
    grid = plot_grid(images, texts=texts, tile_size=args.size)

    if args.file:
        logging.debug('Saving grid image to {}'.format(args.file))
        grid.save(args.file)

    grid.show()

if __name__ == '__main__':
    _main()
