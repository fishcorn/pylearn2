"""
LSH-related costs
"""
__authors__ = "Dustin Webb"
__copyright__ = "Copyright 2014, Universite de Montreal"
__credits__ = ["Dustin Webb"]
__license__ = "3-clause BSD"
__maintainer__ = "Dustin Webb"
__email__ = "pylearn-dev@googlegroups"

from pylearn2.costs.cost import Cost, DefaultDataSpecsMixin
from pylearn2.space import (CompositeSpace, IndexSpace)
from pylearn2.compat import OrderedDict

from theano import tensor as T
from theano import config

import numpy as np


class LSHCriterion(Cost):
    supervised = True

    def __init__(self, m, dim):
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

        # TECHDEBT Fix broadcasting so that the shape fluffing isn't necessary
        l2sq = T.shape_padright(T.sqr(outputs_a - outputs_b).sum(axis = 1))

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

        rval['diff_count'] = difftargets.sum(dtype=config.floatX) 
        rval['like_count'] = liketargets.sum(dtype=config.floatX) 

        # TECHDEBT Fix broadcasting so that the shape fluffing isn't necessary
        l2sq = T.shape_padright(T.sqr(outputs_a - outputs_b).sum(axis = 1))

        likeloss = liketargets * l2sq 
        diffloss = difftargets * T.sqr(T.maximum(0.0, self.m - T.sqrt(l2sq)))

        rval['like_obj'] = likeloss.sum(dtype=config.floatX)
        rval['diff_obj'] = diffloss.sum(dtype=config.floatX)

        # rval["y_misclass"] = T.cast(targets.sum()/targets.shape[0], config.floatX)

        return rval
