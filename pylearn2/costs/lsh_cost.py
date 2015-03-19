"""
LSH-related costs
"""
__authors__ = "Dustin Webb"
__copyright__ = "Copyright 2014, Universite de Montreal"
__credits__ = ["Dustin Webb"]
__license__ = "3-clause BSD"
__maintainer__ = "Dustin Webb"
__email__ = "pylearn-dev@googlegroups"

from pylearn2.costs.cost import Cost
from pylearn2.space import CompositeSpace, IndexSpace
from pylearn2.compat import OrderedDict

from theano import tensor as T
from theano import config

import numpy as np


class LSHCriterion(Cost):
    supervised = True

    def __init__(self, m, dim):
        '''
        m: the threshold that inputs of different classes must be lower
        than in order to affect training.
        dim: must match the number of classes in the data.

        '''
        assert(m > 0)
        self.m = m

        assert(dim > 0)
        self.dim = dim

    def expr(self, model, data, **kwargs):
        space, source = self.get_data_specs(model)
        space.validate(data)

        from theano.printing import Print

        inputs, targets = data
        outputs = model.fprop(inputs)
        batch_size = model.batch_size/2

        outputs_a = outputs[:batch_size]
        outputs_b = outputs[batch_size:]

        targets_a = targets[:batch_size]
        targets_b = targets[batch_size:]

        difftargets = T.neq(targets_a, targets_b)
        liketargets = 1 - difftargets

        l2sq = T.sqr(outputs_a - outputs_b).sum(axis=1, keepdims=True)

        likeloss = liketargets * l2sq 
        diffloss = difftargets * T.sqr(T.maximum(0.0, self.m - T.sqrt(l2sq)))

        loss = (likeloss + diffloss).sum(dtype=config.floatX)

        return loss

    def get_data_specs(self, model):
        return (
            CompositeSpace([
                model.get_input_space(),
                IndexSpace(dim=1, max_labels=self.dim)
            ]),
            (
                model.get_input_source(),
                'targets'
            )
        )

    def get_monitoring_channels(self, model, data, **kwargs):
        rval = OrderedDict()

        space, source = self.get_data_specs(model)
        space.validate(data)

        from theano.printing import Print

        inputs, targets = data
        outputs = model.fprop(inputs)
        batch_size = model.batch_size/2

        outputs_a = outputs[:batch_size]
        outputs_b = outputs[batch_size:]

        targets_a = targets[:batch_size]
        targets_b = targets[batch_size:]

        difftargets = T.neq(targets_a, targets_b)
        liketargets = 1 - difftargets

        l2sq = T.sqr(outputs_a - outputs_b).sum(axis=1, keepdims=True)
        l2 = T.sqrt(l2sq)

        likeloss = liketargets * l2sq 
        diffloss = difftargets * T.sqr(T.maximum(0.0, self.m - l2))

        rval['lsh_like_obj'] = likeloss.sum(dtype=config.floatX)
        rval['lsh_diff_obj'] = diffloss.sum(dtype=config.floatX)

        likesum = liketargets.sum(dtype='int64')
        diffsum = difftargets.sum(dtype='int64')

        hamm = (outputs_a*outputs_b < 0).sum(axis=1, keepdims=True, dtype='int64')
        hmin = hamm.min()
        hmax = hamm.max()

        rval['lsh_like_hamm_min']  = T.switch(liketargets, hamm, hmax).min().astype(config.floatX)
        rval['lsh_like_hamm_mean'] = ((hamm * liketargets).sum(dtype='float64')/likesum).astype(config.floatX)
        rval['lsh_like_hamm_max']  = T.switch(liketargets, hamm, hmin).max().astype(config.floatX)
        rval['lsh_diff_hamm_min']  = T.switch(difftargets, hamm, hmax).min().astype(config.floatX)
        rval['lsh_diff_hamm_mean'] = ((hamm * difftargets).sum(dtype='float64')/diffsum).astype(config.floatX)
        rval['lsh_diff_hamm_max']  = T.switch(difftargets, hamm, hmin).max().astype(config.floatX)

        l2min = l2.min()
        l2max = l2.max()

        rval['lsh_like_l2_min']  = T.switch(liketargets, l2, l2max).min().astype(config.floatX)
        rval['lsh_like_l2_mean'] = ((l2 * liketargets).sum(dtype='float64')/likesum).astype(config.floatX)
        rval['lsh_like_l2_max']  = T.switch(liketargets, l2, l2min).max().astype(config.floatX)
        rval['lsh_diff_l2_min']  = T.switch(difftargets, l2, l2max).min().astype(config.floatX)
        rval['lsh_diff_l2_mean'] = ((l2 * difftargets).sum(dtype='float64')/diffsum).astype(config.floatX)
        rval['lsh_diff_l2_max']  = T.switch(difftargets, l2, l2min).max().astype(config.floatX)

        # Hamming distance is just the size of the symmetric difference
        pos = (outputs > 0).astype('int64')
        pos_sum = pos.sum(axis=1, keepdims=True)
        hamm_mat = pos_sum.T + pos_sum - 2*pos.dot(pos.T)
        # Add max to diagonal so that we don't pick identical points
        hamm_mat += T.identity_like(hamm_mat) * hamm_mat.max()
        hamm_nn = hamm_mat.argsort(axis=0)
        # Compare targets to targets of 7 nearest neighbors
        targ_same = T.eq(targets[hamm_nn[:7]], T.shape_padleft(targets))
        rval['lsh_hamm_nn_match'] = targ_same.sum(axis=0).mean().astype(config.floatX)

        sqr = T.sqr(outputs)
        sqr_sum = sqr.sum(axis=1, keepdims=True)
        l2_mat = sqr_sum.T + sqr_sum - 2*sqr.dot(sqr.T)
        # Add max to diagonal so that we don't pick identical points
        l2_mat += T.identity_like(l2_mat) * l2_mat.max()
        l2_nn = l2_mat.argsort(axis=0)
        # Compare targets to targets of 7 nearest neighbors (also
        # don't bother with taking square root -- we're only comparing
        # and the squared norm will work fine for that)
        targ_same = T.eq(targets[l2_nn[:7]], T.shape_padleft(targets))
        rval['lsh_l2_nn_match'] = targ_same.sum(axis=0).mean().astype(config.floatX)

        return rval
