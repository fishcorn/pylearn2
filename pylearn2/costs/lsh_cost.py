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
import theano

import numpy as np


class LSHCriterion(Cost):
    supervised = True

    def __init__(self, m, dim, interleaved=False):
        '''
        m: the threshold that inputs of different classes must be lower
        than in order to affect training.
        dim: must match the number of classes in the data.
        interleaved: pair data in batch by adjacent pairs (True) or as two contiguous blocks (False)

        '''
        assert(m > 0)
        self.m = m

        assert(dim > 0)
        self.dim = dim

        self.interleaved = interleaved

    def expr(self, model, data, **kwargs):
        space, source = self.get_data_specs(model)
        space.validate(data)

        from theano.printing import Print

        inputs, targets = data
        outputs = model.fprop(inputs)
        batch_size = model.batch_size/2

        if self.interleaved: 
            diffoutputs = outputs[0::2] - outputs[1::2]
            difftargets = T.neq(targets[0::2], targets[1::2])
        else:
            diffoutputs = outputs[:batch_size] - outputs[batch_size:]
            difftargets = T.neq(targets[:batch_size], targets[batch_size:])

        l2sq = T.sqr(diffoutputs).sum(axis=1, dtype=config.floatX, keepdims=True)
        # Use this augmented norm becuase Theano isn't smart enough not to div by zero
        mmnsq = T.sqr(self.m - T.sqrt(.0001 + l2sq))

        loss = T.switch(difftargets, mmnsq, l2sq).sum(dtype=config.floatX)

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

        if self.interleaved: 
            diffoutputs = outputs[0::2] - outputs[1::2]
            difftargets = T.neq(targets[0::2], targets[1::2])
        else:
            diffoutputs = outputs[:batch_size] - outputs[batch_size:]
            difftargets = T.neq(targets[:batch_size], targets[batch_size:])
        liketargets = 1 - difftargets

        l2sq = T.sqr(diffoutputs).sum(axis=1, dtype=config.floatX, keepdims=True)
        l2 = T.sqrt(l2sq)
        # Use this augmented norm becuase Theano isn't smart enough not to div by zero
        mmnsq = T.sqr(self.m - T.sqrt(.0001 + l2sq))

        likeloss = T.switch(liketargets, l2sq, 0.)
        diffloss = T.switch(difftargets, mmnsq, 0.) 

        rval['lsh_like_obj'] = likeloss.sum(dtype=config.floatX)
        rval['lsh_diff_obj'] = diffloss.sum(dtype=config.floatX)

        likesum = liketargets.sum(dtype=config.floatX)
        diffsum = difftargets.sum(dtype=config.floatX)

        if self.interleaved:
            hamm = outputs[0::2] * outputs[1::2] < 0
        else:
            hamm = outputs[:batch_size] * outputs[batch_size:] < 0
        hamm = hamm.sum(axis=1, keepdims=True, dtype=config.floatX)
        hmin = hamm.min()
        hmax = hamm.max()

        rval['lsh_like_hamm_min']  = T.switch(liketargets, hamm, hmax).min()
        rval['lsh_like_hamm_mean'] = (hamm * liketargets).sum(dtype=config.floatX)/likesum
        rval['lsh_like_hamm_max']  = T.switch(liketargets, hamm, hmin).max()
        rval['lsh_diff_hamm_min']  = T.switch(difftargets, hamm, hmax).min()
        rval['lsh_diff_hamm_mean'] = (hamm * difftargets).sum(dtype=config.floatX)/diffsum
        rval['lsh_diff_hamm_max']  = T.switch(difftargets, hamm, hmin).max()

        l2min = l2.min()
        l2max = l2.max()

        rval['lsh_like_l2_min']  = T.switch(liketargets, l2, l2max).min() 
        rval['lsh_like_l2_mean'] = (l2 * liketargets).sum(dtype=config.floatX)/likesum
        rval['lsh_like_l2_max']  = T.switch(liketargets, l2, l2min).max() 
        rval['lsh_diff_l2_min']  = T.switch(difftargets, l2, l2max).min() 
        rval['lsh_diff_l2_mean'] = (l2 * difftargets).sum(dtype=config.floatX)/diffsum
        rval['lsh_diff_l2_max']  = T.switch(difftargets, l2, l2min).max() 

        # Hamming distance is just the size of the symmetric difference
        pos = (outputs > 0).astype('int32')
        pos_sum = pos.sum(axis=1, keepdims=True)
        hamm_mat = pos_sum.T + pos_sum - 2*pos.dot(pos.T)
        # Add max to diagonal so that we don't pick identical points
        hamm_mat += T.identity_like(hamm_mat) * (hamm_mat.max() + 1)
        hamm_nn = hamm_mat.argsort(axis=0)[:-1,:]
        # Compare targets to targets of 7 nearest neighbors
        targ_pred = T.eq(targets[hamm_nn], T.shape_padleft(targets))
        rval['lsh_hamm_nn_match'] = targ_pred[:7].sum(axis=0).mean(dtype=config.floatX)

        # Compute mean average precision over all points
        batch_range = np.arange(1, model.batch_size, dtype=config.floatX)
        # Average over all query points
        allpred = targ_pred.mean(axis=1, dtype=config.floatX).flatten().cumsum()
        # Compute (inf. retrieval) precision and recall
        rec = allpred/allpred[-1]
        prc = allpred/batch_range
        # Sort first by precision and then those by recall (idx1[idx2] is the resulting order)
        idx1 = prc.argsort(kind='mergesort').flatten()[::-1]
        idx2 = rec[idx1].argsort(kind='mergesort').flatten()
        # Condition for integration and compute MAP
        rec = T.concatenate([np.array([0.],dtype=config.floatX),rec[idx1[idx2]].flatten()])
        prc = T.concatenate([np.array([1.],dtype=config.floatX),prc[idx1[idx2]].flatten()])
        widths = T.extra_ops.diff(rec)
        heights = prc[:-1] + prc[1:]
        rval['lsh_hamm_nn_map'] = widths.dot(heights)/2 # Trapezoid method

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
        rval['lsh_l2_nn_match'] = targ_same.sum(axis=0).mean(dtype=config.floatX)

        return rval

# vim: tabstop=8 expandtab shiftwidth=4 softtabstop=4
