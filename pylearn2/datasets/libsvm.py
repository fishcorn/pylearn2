"""
The LibSVMDataset class loads datasets found on libsvm website into a
DenseDesignMatrix.
"""
__authors__ = "Dustin Webb"
__copyright__ = "Copyright 2010-2012, Universite de Montreal"
__credits__ = ["Dustin Webb"]
__license__ = "3-clause BSD"
__maintainer__ = "LISA Lab"
__email__ = "pylearn-dev@googlegroups"
import os

import numpy
from sklearn.datasets import load_svmlight_file
from sklearn import preprocessing
from sklearn.preprocessing import scale

from pylearn2.utils.string_utils import preprocess
from dense_design_matrix import DenseDesignMatrix
from pylearn2.format.target_format import convert_to_one_hot

import theano

class LibSVMDataset(DenseDesignMatrix):
    def __init__(self, dataset_name, which_set, split=None,
                 use_test_set=False, randomize_first=True, use_scaled=True,
                 rng_seed=0):
        """
        Creates an LibSVMDataset object.

        Parameters
        ----------
        file : str
            The name of a libsvm dataset.
        split : 3-tuple
            A 3 element tuple where the elements sum to 1. The elements indicate
            the percentage samples to be used for training, validation, and
            testing respectively.
        """
        self.dataset_name = dataset_name

        assert(which_set in ['train', 'valid', 'test'])
        self.which_set = which_set

        # Validate split before loading data to avoid costly loading time if
        # there is a problem
        if split is not None:
            pass
        elif use_test_set:
            # If we're using the test set then we only need to split the traing
            # set in to training and validation compoenents.
            split = (0.75, 0.25, 0.0)
        else:
            # If we're not using the test set then we need to split the training
            # set in to training, validation, and testing components.
            split = (0.6, 0.15, 0.25)
        assert(len(split) == 3)
        assert(sum(split) == 1.0)
        assert(all(i > 0.0 and i < 1.0 for i in split))
        self.split = split
    
        base_path = os.path.join(
            preprocess("${PYLEARN2_DATA_PATH}"),
            self.dataset_name
            )

        if use_scaled:
            path = base_path + '_scale'
            if os.path.exists(path):
                base_path = path
            else:
                print '\'%s\' does not exist. Defaulting to \'%s\'.' % (
                    path,
                    base_path
                    )

        if use_test_set:
            base_path += '.t'

        assert(os.path.exists(base_path))
        self.base_path = base_path

        if use_test_set:
            X, y = load_svm_light_file(self.base_path)
        else:    
            X, y = self.load_svmlight_file(self.base_path)

            if randomize_first:
                rng = numpy.random.RandomState(rng_seed)
                permutation = rng.permutation(range(X.shape[0]))
                X = X[permutation]
                y = y[permutation]
    
            train_samples = int(numpy.floor(self.split[0]*X.shape[0]))
            valid_samples = int(numpy.ceil(self.split[1]*X.shape[0]))
            test_samples = X.shape[0] - train_samples - valid_samples

            train_X = X[:train_samples]
            train_y = y[:train_samples]

            end_idx = train_samples + valid_samples
            valid_X = X[train_samples:end_idx]
            valid_y = y[train_samples:end_idx]

            start_idx = train_samples + valid_samples
            test_X = X[start_idx:]
            test_y = y[start_idx:]

            if which_set == 'train':
                X = train_X
                y = train_y
            elif which_set == 'valid':
                X = valid_X
                y = valid_y
            else:
                X = test_X
                y = test_y

            X = X.astype('float32')
    
        super(LibSVMDataset, self).__init__(X=X, y=y)

    def load_svmlight_file(self, file_path):
        X, y = load_svmlight_file(file_path)

        X = X.todense().astype(theano.config.floatX)
        #X = scale(X)

        x = numpy.zeros(y.shape)
        unique_vals = numpy.unique(y)
        num_unique_vals = len(unique_vals)
        for idx, val in enumerate(unique_vals):
            x[y == val] = idx
        x = x.astype('int32')
        y = convert_to_one_hot(x, max_labels=num_unique_vals)
        #y = y.astype('int32')
        #y = y + numpy.abs(numpy.min(y))
        #y = y.reshape(y.shape[0], 1).astype(theano.config.floatX)

        return (X, y)

