#!/usr/bin/env python
from __future__ import print_function, division
import argparse
import copy
import logging
import math
import sys
if sys.version_info[0] < 3:
    import cPickle as pickle
else:
    import pickle
import os
import sklearn.cluster
import sklearn.metrics
import collections
import gui

import plot_grid


CACHE_DIR='./cache.tmp'

def create_grid(images, features, n_clusters):
    if not os.path.exists(CACHE_DIR):
        os.makedirs(CACHE_DIR)
    imstr = "".join(sorted(images))
    path = os.path.join(CACHE_DIR, '{}_{}_{}.p'.format(n_clusters, len(images), hex(abs(hash(imstr)))[2:]))
    if os.path.exists(path):
        membership, medoids, grid = pickle.load(open(path, 'rb'))
        logging.debug('Loaded {}'.format(path))
    else:
        if len(images) < n_clusters:
            medoids = [x for x in range(len(images))]
            membership = []
        else:
            membership, medoids = do_kmeans(features, n_clusters)
        text = [str(membership.tolist().count(i)) for i,_ in enumerate(medoids)] if membership != [] else None
        grid = plot_grid.plot_grid([images[x] for x in medoids], text, 200)
        pickle.dump([membership, medoids, grid], open(path, 'wb'))
        logging.debug('Wrote {}'.format(path))

    return membership, medoids, grid

def do_kmeans(features, n_clusters):
    logging.debug('Running kmeans. len(features)={}, n_clusters={}'.format(len(features), n_clusters))
    kmeans = sklearn.cluster.KMeans(n_clusters=n_clusters, random_state=0).fit(features)
    membership = kmeans.predict(features)
    class_sizes = [membership.tolist().count(x) for x in set(membership)]
    logging.debug('Class sizes: smallest: {}, largest: {}'.format(min(class_sizes), max(class_sizes)))
    medoids = []
    logging.debug('Looking for medoids in {} classes'.format(n_clusters))
    for class_ in range(n_clusters):
        class_members = [i for i, c in enumerate(membership) if c == class_]
        class_features = [features[i] for i in class_members]
        if class_members != []:
            medoid, _ = sklearn.metrics.pairwise_distances_argmin_min([kmeans.cluster_centers_[class_]], class_features)
            medoids.append(class_members[medoid[0]])
        else:
            logging.warning('Class {} has no members. Duplicates?'.format(class_))
    assert [k for k,v in collections.Counter(medoids).items() if v>1] == [], "duplicates in medoids"
    return membership, medoids

def interactive(images, pca_features, n_clusters):
    history = []
    while True:
        nxy = int(math.ceil(math.sqrt(min(len(images), n_clusters))))
        logging.debug("len(images)={} nxy={}".format(len(images), nxy))
        membership, medoids, grid = create_grid(images=images, features=pca_features, n_clusters=nxy*nxy)
        ok, x, y = gui.select_square(grid, nxy, nxy)
        index = x + y*nxy
        if ok and index < len(medoids):
            logging.info("Selected {}".format(images[medoids[index]]))
            if membership == [] or len(medoids) == 1:
                logging.debug("Done")
                return images[medoids[index]]
            selected = [i for i,c in enumerate(membership) if c == index]
            logging.debug("Selected {},{} => index={} len(selected)={}".format(x, y, index, len(selected)))
            history.append((copy.copy(images), copy.copy(pca_features)))
            images = [images[i] for i in selected]
            pca_features = [pca_features[i] for i in selected]
        else:
            if history == []:
                return None
            (images, pca_features) = history[-1]
            history = history[:-1]

def parse_arguments():
    parser = argparse.ArgumentParser(description='Classify images using K-means',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-v', '--verbose', action='count', default=0)
    parser.add_argument('-p', '--pca_features_file', default='pca_features.p',
                        help='The PCA features file name')
    parser.add_argument('-f', '--file', default='grid.jpg',
                        help='Resulting grid image file name')   
    parser.add_argument('-n', '--n_clusters', type=int, default=25,
                        help='Number of clusters')
    parser.add_argument('-s', '--size', type=int, default=160,
                        help='Size of each tile')
    parser.add_argument('-i', '--interactive', action='store_true',
                        help='Enter interactive mode')
    

    args = parser.parse_args()
    return args

def _main():
    args = parse_arguments()
    level = (logging.WARNING, logging.INFO, logging.DEBUG)[min(args.verbose, 2)]
    logging.basicConfig(level=level)

    images, pca_features = pickle.load(open(args.pca_features_file, 'rb'))

    if args.interactive:
        selected = interactive(images, pca_features, n_clusters = args.n_clusters)
        if selected is not None:
            print("{}".format(selected))
    else:
        membership, medoids = do_kmeans(features=pca_features, n_clusters=args.n_clusters)
        texts = ["{} {} {}".format(i, membership.tolist().count(i), os.path.basename(images[x])) for i, x in enumerate(medoids)]
        grid = plot_grid.plot_grid([images[x] for x in medoids], texts, args.size)

        logging.debug('Saving grid image to {}'.format(args.file))
        grid.save(args.file)
        grid.show()

if __name__ == '__main__':
    _main()
