#!/usr/bin/env python
from __future__ import print_function
import argparse
import logging
import math
import os
import PIL
from PIL import ImageFont
from PIL import ImageDraw


def plot_grid(images, texts=None, tile_size=120):
    font_height = 8
    font = PIL.ImageFont.truetype("/usr/share/fonts/truetype/ubuntu-font-family/Ubuntu-B.ttf", font_height)
    n = int(math.ceil(math.sqrt(len(images))))
    full_size = tile_size * n
    grid_image = PIL.Image.new('RGB', (full_size, full_size))
    for i, img in enumerate(images):
        x, y = (i % n) * tile_size, (i / n) * tile_size
        tile = PIL.Image.open(img)
        margin = abs((tile.width - tile.height) / 2)
        if (tile.width > tile.height):
            tile = tile.crop((margin, 0, margin + tile.height, tile.height))
        else:
            tile = tile.crop((0, margin, tile.width, margin + tile.width))
        tile = tile.resize((tile_size, tile_size), PIL.Image.ANTIALIAS)
        draw = PIL.ImageDraw.Draw(tile)
        if texts:
            draw.text((0, tile_size - font_height), texts[i], (255, 255, 255), font=font)
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
       
    grid = plot_grid(images, texts=[os.path.basename(f) for f in images], tile_size=args.size)

    if args.file:
        logging.debug('Saving grid image to {}'.format(args.file))
        grid.save(args.file)

    grid.show()

if __name__ == '__main__':
    _main()
