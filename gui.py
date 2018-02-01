#!/usr/bin/env python
import argparse
import logging
import Tkinter as tk
from PIL import Image, ImageTk

def select_square(image, nx, ny):
    def __handle_click(event, xy):
        xy.append(event.x)
        xy.append(event.y)    
        event.widget.quit()

    window = tk.Tk()
    xy = []
    window.bind('<Button-1>', lambda e: __handle_click(e, xy))

    size = int(min(window.winfo_screenwidth(), window.winfo_screenheight()) * 0.9)
    image = image.resize((size, size), Image.ANTIALIAS)

    tk_image = ImageTk.PhotoImage(image)
    panel = tk.Label(window, image = tk_image)
    panel.pack(side = "bottom", fill = "both", expand = "yes")
    window.mainloop()
    logging.debug("Clicking at position {}".format(xy))    
    x, y = xy
    width, height = image.size
    window.destroy()
    return x / (width / nx), y / (height / ny)

def parse_arguments():
    parser = argparse.ArgumentParser(description='Show picture grid',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-v', '--verbose', action='count')
    parser.add_argument('-f', '--file', default='grid.jpg',
                        help='Input  grid image file name')   
    parser.add_argument('--nx', type=int, default=10,
                        help='X dimension')
    parser.add_argument('--ny', type=int, default=10,
                        help='Y dimension')
    args = parser.parse_args()
    return args


def _main():
    args = parse_arguments()
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=level)
    
    x, y = select_square(Image.open(args.file), args.nx, args.ny)
    print(x, y)


if __name__ == '__main__':
    _main()
