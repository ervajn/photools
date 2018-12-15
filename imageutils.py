import os
import PIL
import rawpy

IMAGE_RAW_TYPES = ['.raw','.nef','.orf']
IMAGE_TYPES = ['.jpg','.png','.jpeg'] + IMAGE_RAW_TYPES

def load_image(path, target_size=None):
    if os.path.splitext(path)[1].lower() in IMAGE_RAW_TYPES:
        rgb = rawpy.imread(path).postprocess()
        img = PIL.Image.fromarray(rgb)
    else:
        img = PIL.Image.open(path)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    if target_size is not None:
        hw_tuple = (target_size[1], target_size[0])
        if img.size != hw_tuple:
            img = img.resize(hw_tuple)
    return img
