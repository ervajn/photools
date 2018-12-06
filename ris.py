#!/usr/bin/env python
"""
https://github.com/ml4a/ml4a-guides/blob/master/notebooks/image-search.ipynb
https://github.com/ml4a/ml4a-guides/blob/master/notebooks/image-tsne.ipynb
"""
import argparse
import logging
import math
import fnmatch
import os
import random
import sys
if sys.version_info[0] < 3:
    import cPickle as pickle
    import rasterfairy
else:
    import pickle
import csv
import numpy as np
import matplotlib.pyplot
import keras.preprocessing.image
import keras.applications.imagenet_utils
import keras.models
import sklearn.decomposition
import scipy.spatial
import tqdm
import PIL
import sklearn.manifold

def write_csv(images, features):
    with open('image_features.csv', 'wb') as csvfile:
        csvwriter = csv.writer(csvfile, quoting=csv.QUOTE_MINIMAL)
        for i, f in zip(images, features):
            csvwriter.writerow([i] + f.tolist())

def get_image(path, target_size):
    try:
        img = keras.preprocessing.image.load_img(path, target_size=target_size)
        x = keras.preprocessing.image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = keras.applications.imagenet_utils.preprocess_input(x)
        return img, x
    except:
        logging.warning("Failed to handle {}".format(path))
        return None, None

def get_feature_extractor():
    model = keras.applications.VGG16(weights='imagenet', include_top=True)
    #model = keras.applications.VGG19(weights='imagenet', include_top=True)
    logging.debug("Output layer: {}".format(model.get_layer("fc2").output))
    feat_extractor = keras.models.Model(inputs=model.input, outputs=model.get_layer("fc2").output)
    return feat_extractor

def find_images(images_path, max_num_images=0):
    images = [os.path.join(dp, f) for dp, dn, filenames in
              os.walk(images_path) for f in filenames if os.path.splitext(f)[1].lower() in ['.jpg','.png','.jpeg']]
    if max_num_images > 0 and max_num_images < len(images):
        images = [images[i] for i in sorted(random.sample(range(len(images)), max_num_images))]
    logging.debug('Using {} images from {}'.format(len(images), images_path))
    return images

def _extract_pca_features(feat_extractor, images, n_components=300):
    logging.debug('Extracting features from {} images'.format(len(images)))
    result_images = []
    features = []
    for image_path in tqdm.tqdm(images):
        img, x = get_image(image_path, target_size=feat_extractor.input_shape[1:3])
        if img:
            result_images.append(image_path)
            feat = feat_extractor.predict(x)[0]
            features.append(feat)

    #pickle.dump([images, features], open('image_features.p', 'wb'))
    #write_csv(images, features)

    logging.debug('Performing PCA. reducing {} -> {}'.format(len(features[0]), n_components))
    features = np.array(features)
    pca = sklearn.decomposition.PCA(n_components=n_components)
    pca.fit(features)

    #pickle.dump([pca], open('pca_model.p', 'wb'))

    pca_features = pca.transform(features)
    return result_images, pca_features

def get_pca_features(images_path, pca_features_file):
    if os.path.exists(pca_features_file):
        logging.info('Features file already exists. Loading {} and ignoring images path "{}"'.
                     format(pca_features_file, images_path))
        images, pca_features = pickle.load(open(pca_features_file, 'rb'))
    else:
        images = find_images(images_path)
        if not images:
            logging.error('No images found in {}'.format(images_path))
            exit(-1)    
        feat_extractor = get_feature_extractor()
        images, pca_features = _extract_pca_features(feat_extractor, images)
        pickle.dump([images, pca_features], open(pca_features_file, 'wb'))
    return images, pca_features    

def get_closest_images(pca_features, query_image_idx, num_results=5):
    distances = [scipy.spatial.distance.euclidean(pca_features[query_image_idx], feat) for feat in pca_features ]
    idx_closest = sorted(range(len(distances)), key=lambda k: distances[k])[1:num_results+1]
    return idx_closest

def get_concatenated_images(images, indexes, thumb_height):
    border = PIL.Image.fromarray(np.full([thumb_height, thumb_height/10, 3], fill_value=255, dtype=np.int8), 'RGB')
    thumbs = []
    for idx in indexes:
        img = keras.preprocessing.image.load_img(images[idx])
        img = img.resize((int(img.width * thumb_height / img.height), thumb_height))
        thumbs.append(img)
        if len(thumbs) == 1:
            thumbs.append(border)
    concat_image = np.concatenate([np.asarray(t) for t in thumbs], axis=1)
    return concat_image

def show_closest_images(pca_features, images, query_image_idx):
    idx_closest = get_closest_images(pca_features, query_image_idx)
    results_image = get_concatenated_images(images, [query_image_idx] + idx_closest, 300)
    
    logging.debug('Closest to {}: {}'.format(images[query_image_idx], [images[x] for x in idx_closest]))

    print('Closest to {}'.format(images[query_image_idx]))
    for x in idx_closest:
        print('  {}'.format(images[x]))

    return results_image

def get_file_index(images, pattern):
    if pattern.isdigit() and int(pattern) < len(images):
        return int(pattern)
    for i, path in enumerate(images):
        if fnmatch.fnmatch(path, pattern):
            return i
    logging.error('Found no image matching {}'.format(pattern))
    return None

def calculate_tsne(images, pca_features, num_images_to_use=4*256):
    if len(images) > num_images_to_use:
        sort_order = sorted(random.sample(range(len(images)), num_images_to_use))
        images = [images[i] for i in sort_order]
        pca_features = [pca_features[i] for i in sort_order]

    X = np.array(pca_features)
    tsne = sklearn.manifold.TSNE(n_components=2, learning_rate=150, perplexity=30, angle=0.2, verbose=2).fit_transform(X)

    tx, ty = tsne[:,0], tsne[:,1]
    tx = (tx-np.min(tx)) / (np.max(tx) - np.min(tx))
    ty = (ty-np.min(ty)) / (np.max(ty) - np.min(ty))

    return images, tsne, tx, ty

def plot_tsne(images, tsne, tx, ty):
    width = 4000
    height = 3000
    max_dim = 100

    full_image = PIL.Image.new('RGBA', (width, height))
    for img, x, y in tqdm.tqdm(zip(images, tx, ty)):
        tile = PIL.Image.open(img)
        rs = max(1, tile.width/max_dim, tile.height/max_dim)
        tile = tile.resize((int(tile.width/rs), int(tile.height/rs)), PIL.Image.ANTIALIAS)
        full_image.paste(tile, (int((width-max_dim)*x), int((height-max_dim)*y)), mask=tile.convert('RGBA'))

    return full_image

def plot_grid(images, tsne, tx, ty):
    nx = int(math.ceil(math.sqrt(len(images))))
    ny = nx

    grid_assignment = rasterfairy.transformPointCloud2D(tsne, target=(nx, ny))

    tile_width = 144
    tile_height = 112

    full_width = tile_width * nx
    full_height = tile_height * ny
    aspect_ratio = float(tile_width) / tile_height

    grid_image = PIL.Image.new('RGB', (full_width, full_height))

    for img, grid_pos in tqdm.tqdm(zip(images, grid_assignment[0].tolist())):
        idx_x, idx_y = grid_pos
        x, y = tile_width * idx_x, tile_height * idx_y
        tile = PIL.Image.open(img)
        tile_ar = float(tile.width) / tile.height  # center-crop the tile to match aspect_ratio
        if (tile_ar > aspect_ratio):
            margin = 0.5 * (tile.width - aspect_ratio * tile.height)
            tile = tile.crop((margin, 0, margin + aspect_ratio * tile.height, tile.height))
        else:
            margin = 0.5 * (tile.height - float(tile.width) / aspect_ratio)
            tile = tile.crop((0, margin, tile.width, margin + float(tile.width) / aspect_ratio))
        tile = tile.resize((tile_width, tile_height), PIL.Image.ANTIALIAS)
        grid_image.paste(tile, (int(x), int(y)))

    return grid_image

def do_sne(images, pca_features, filenamebase):
    images, tsne, tx, ty = calculate_tsne(images, pca_features)
    tsne_image = plot_tsne(images, tsne, tx, ty)
    tsne_image = tsne_image.convert('RGB')
    tsne_image.save(filenamebase + '_tsne.jpg')
    if sys.version_info[0] < 3:
        grid_image = plot_grid(images, tsne, tx, ty)
        grid_image.save(filenamebase + '_grid.jpg')
    else:
        logging.info('No grid image since rasterfairy is not available for Pytyhon3')
        grid_image = None
    return [tsne_image, grid_image] if grid_image else [tsne_image]

def show_images(images):
    for i in images:
        matplotlib.pyplot.figure(figsize = (16,12))
        matplotlib.pyplot.imshow(i)
    matplotlib.pyplot.show()        

def parse_arguments():
    parser = argparse.ArgumentParser(description='Calculate image features',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-v', '--verbose', action='count')
    parser.add_argument('-i', '--images_path', default='.',
                        help='The directory with photos')
    parser.add_argument('-p', '--pca_features_file', default='pca_features.p',
                        help='The PCA features file name')
    parser.add_argument('-x', '--index', default='',
                        help='Find closest images to this image')

    parser.add_argument('-t', '--tsne', default='',
                        help='Filename to save TSNE image')

    parser.add_argument('-s', '--show', action='store_true',
                        help='Show the resulting images')
    
    args = parser.parse_args()   
    return args

def _main():
    args = parse_arguments()
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=level)
    
    images, pca_features = get_pca_features(args.images_path, args.pca_features_file)
    images_to_show = []
    
    if args.index:
        ix = get_file_index(images, args.index)
        if ix is not None:
            images_to_show.append(show_closest_images(pca_features, images, ix))

    if args.tsne:
        ims = do_sne(images, pca_features, args.tsne)
        images_to_show += ims

    if args.show:
        show_images(images_to_show)
            
if __name__ == '__main__':
    _main()
