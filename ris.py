#!/usr/bin/env python
"""
Reverse image search tool
https://github.com/ml4a/ml4a-guides/blob/f31929024c51bdc409b52e91a77725291fa16564/notebooks/image-search.ipynb
"""
import argparse
import logging
import os
import random
import cPickle as pickle
import numpy as np
import matplotlib.pyplot
from matplotlib.pyplot import imshow
import keras
from keras.preprocessing import image
from keras.applications.imagenet_utils import decode_predictions, preprocess_input
from keras.models import Model
from sklearn.decomposition import PCA
from scipy.spatial import distance
from tqdm import tqdm

logger = logging.getLogger(__name__)


def get_image(path, target_size):
    img = image.load_img(path, target_size=target_size)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return img, x

def get_feature_extractor():
    model = keras.applications.VGG16(weights='imagenet', include_top=True)   
    feat_extractor = Model(inputs=model.input, outputs=model.get_layer("fc2").output)
    return feat_extractor

def find_images(images_path, max_num_images=0):
    images = [os.path.join(dp, f) for dp, dn, filenames in
              os.walk(images_path) for f in filenames if os.path.splitext(f)[1].lower() in ['.jpg','.png','.jpeg']]
    if max_num_images > 0 and max_num_images < len(images):
        images = [images[i] for i in sorted(random.sample(xrange(len(images)), max_num_images))]
    logger.debug('Using {} images from {}'.format(len(images), images_path))
    return images

def _extract_pca_features(feat_extractor, images, n_components=300):
    features = []
    for image_path in tqdm(images):
        img, x = get_image(image_path, target_size=feat_extractor.input_shape[1:3])
        feat = feat_extractor.predict(x)[0]
        features.append(feat)

    features = np.array(features)
    pca = PCA(n_components=n_components)
    pca.fit(features)
    pca_features = pca.transform(features)
    return pca_features

def get_pca_features(images_path, pca_features_file):
    if os.path.exists(pca_features_file):
        logger.debug('Features file already exists. Loading {} and ignoring images path {}'.
                     format(pca_features_file, pca_features_file, images_path))
        images, pca_features = pickle.load(open(pca_features_file, 'r'))
    else:
        images = find_images(images_path)
        if not images:
            logger.error('No images found in {}'.format(images_path))
            exit(-1)    
        feat_extractor = get_feature_extractor()
        pca_features = _extract_pca_features(feat_extractor, images)
        pickle.dump([images, pca_features], open(pca_features_file, 'wb'))
    return images, pca_features    

def get_closest_images(pca_features, query_image_idx, num_results=5):
    distances = [distance.euclidean(pca_features[query_image_idx], feat) for feat in pca_features ]
    idx_closest = sorted(range(len(distances)), key=lambda k: distances[k])[1:num_results+1]
    return idx_closest

def get_concatenated_images(images, indexes, thumb_height):
    thumbs = []
    for idx in indexes:
        img = image.load_img(images[idx])
        img = img.resize((int(img.width * thumb_height / img.height), thumb_height))
        thumbs.append(img)
    concat_image = np.concatenate([np.asarray(t) for t in thumbs], axis=1)
    return concat_image

def show_closest_images(pca_features, images, query_image_idx):
    idx_closest = get_closest_images(pca_features, query_image_idx)
    query_image = get_concatenated_images(images, [query_image_idx], 300)
    results_image = get_concatenated_images(images, idx_closest, 200)
    
    logger.debug('Closest to {}: {}'.format(images[query_image_idx], [images[x] + '\n' for x in idx_closest]))

    # display the query image
    matplotlib.pyplot.figure(figsize = (5,5))
    imshow(query_image)
    matplotlib.pyplot.title("query image (%d)" % query_image_idx)

    # display the resulting images
    matplotlib.pyplot.figure(figsize = (16,12))
    imshow(results_image)
    matplotlib.pyplot.title("result images")

    matplotlib.pyplot.show()

def parse_arguments():
    parser = argparse.ArgumentParser(description='Calculate image features')
    parser.add_argument('-v', '--verbose', action='count')
    parser.add_argument('-i', '--images_path', default='.',
                        help='The directory with photos')
    parser.add_argument('-p', '--pca_features_file', default='pca_features.p',
                        help='The PCA features file name')
    parser.add_argument('-x', '--index', default='',
                        help='Find closest images to this image')    
    args = parser.parse_args()   
    return args

def _main():
    args = parse_arguments()
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=level)
    
    images, pca_features = get_pca_features(args.images_path, args.pca_features_file)

    if args.index:
        show_closest_images(pca_features, images, int(args.index))
    
                      
if __name__ == '__main__':
    _main()
