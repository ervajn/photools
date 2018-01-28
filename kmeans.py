#!/usr/bin/env python
from __future__ import print_function
import argparse
import logging
import cPickle as pickle
import os
import sklearn.cluster
import sklearn.metrics
import collections

import plot_grid


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
        medoid, _ = sklearn.metrics.pairwise_distances_argmin_min([kmeans.cluster_centers_[class_]], class_features)
        medoids.append(class_members[medoid[0]])
    assert [k for k,v in collections.Counter(medoids).items() if v>1] == [], "duplicates in medoids"
    return membership, medoids

def parse_arguments():
    parser = argparse.ArgumentParser(description='Classify images using K-means',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-v', '--verbose', action='count')
    parser.add_argument('-p', '--pca_features_file', default='pca_features.p',
                        help='The PCA features file name')
    parser.add_argument('-f', '--file', default='grid.jpg',
                        help='Resulting grid image file name')   
    parser.add_argument('-n', '--n_clusters', type=int, default=100,
                        help='Number of clusters')
    parser.add_argument('-s', '--size', type=int, default=160,
                        help='Size of each tile')

    args = parser.parse_args()
    return args

def _main():
    args = parse_arguments()
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=level)

    images, pca_features = pickle.load(open(args.pca_features_file, 'r'))

    membership, medoids = do_kmeans(features=pca_features, n_clusters=args.n_clusters)

    texts = ["{} {} {}".format(i, membership.tolist().count(i), os.path.basename(images[x])) for i, x in enumerate(medoids)]
    grid = plot_grid.plot_grid([images[x] for x in medoids], texts, args.size)

    logging.debug('Saving grid image to {}'.format(args.file))
    grid.save(args.file)
    grid.show()

if __name__ == '__main__':
    _main()
