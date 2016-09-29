import numpy
import pylearn2.utils.string_utils as su
from pylearn2.datasets.dense_design_matrix import DenseDesignMatrix
from pylearn2.format.target_format import convert_to_one_hot
from sklearn import preprocessing


class YearPredictionMSD(DenseDesignMatrix):
    def __init__(self, which_set, split=(0.8, 0.1, 0.1), randomize_first=False, rng_seed=0, whiten=False):
        assert(which_set in ['train', 'valid', 'test'])
        self.which_set = which_set

        # Validate split before loading data to avoid costly loading time if
        # there is a problem
        assert(len(split) == 3)
        assert(sum(split) == 1.0)
        assert(all(i > 0.0 and i < 1.0 for i in split))
        self.split = split

        raw_data = numpy.genfromtxt(su.preprocess('${PYLEARN2_DATA_PATH}/YearPredictionMSD.txt'),delimiter=',')

        if randomize_first > 0:
            rng = numpy.random.RandomState(seed=rng_seed)
            raw_data = raw_data[rng.permutation(raw_data.shape[0])]

        y = raw_data[:, 0]
        y = y.reshape(y.shape[0], 1)
        y[y<2001] = 0
        y[y>2000] = 1
        X = raw_data[:, 1:]

        train_samples = int(numpy.floor(self.split[0]*X.shape[0]))
        valid_samples = int(numpy.ceil(self.split[1]*X.shape[0]))
        test_samples = X.shape[0] - train_samples - valid_samples

        valid_end_idx = train_samples + valid_samples

        train_X = X[:train_samples]
        train_y = y[:train_samples]

        if whiten:
            scaler = preprocessing.StandardScaler().fit(X)

        if which_set == 'train':
            X = train_X
            y = train_y
        elif which_set == 'valid':
            X = X[train_samples:valid_end_idx]
            y = y[train_samples:valid_end_idx]
        else:
            start_idx = train_samples + valid_samples
            X = X[valid_end_idx:]
            y = y[valid_end_idx:]

        if whiten:
            X = scaler.transform(X)

        X = X.astype('float32')

        y = convert_to_one_hot(y.astype('uint'))
        y = y.reshape(y.shape[0],2)

        return super(YearPredictionMSD, self).__init__(X=X, y=y)
