# Author: Thang Vu
# Date: 25/Nove/2017
# Description: Load datasets

import gzip
from six.moves import cPickle as pickle
import os
import platform


# load pickle based on python version 2 or 3
def load_pickle(f):
    version = platform.python_version_tuple()
    if version[0] == '2':
        return pickle.load(f)
    elif version[0] == '3':
        return pickle.load(f, encoding='latin1')
    raise ValueError("invalid python version: {}".format(version))


def load_mnist_datasets(path='data/mnist.pkl.gz'):
    if not os.path.exists(path):
        raise Exception('Cannot find %s' % path)
    with gzip.open(path, 'rb') as f:
        train_set, val_set, test_set = load_pickle(f)
        return train_set, val_set, test_set
