# --------------------------------------------------------
# Copyright (c) 2017 Microsoft
# --------------------------------------------------------

from cntk import output_variable
from cntk.ops.functions import UserFunction
import numpy as np
import scipy as sp
from fast_rcnn.config import cfg

DEBUG = True

class BinaryLogLossWithIgnore(UserFunction):
    """
    Computes a log loss for two classes and allows to ignore predictions
    """

    def __init__(self, arg1, arg2, name='BinaryLogLossWithIgnore', ignore_label=None):
        super(BinaryLogLossWithIgnore, self).__init__([arg1, arg2], name=name)

        self._ignore_label = ignore_label

    def infer_outputs(self):
        return [output_variable((1), self.inputs[0].dtype, self.inputs[0].dynamic_axes)]

    def forward(self, arguments, device=None, outputs_to_retain=None):
        # Algorithm:
        #
        # skip those where target == ignore_label
        # for the rest:
        #
        # loss = - 1/N sum_{i=1}^N (y_i log(p_i) + (1 - y_i) log(1 - p_i))

        predictions = arguments[0][0,0,:]
        targets = arguments[1][0,0,:]

        #import pdb; pdb.set_trace()

        targets_flat = targets.flatten()
        pred_flat = predictions.flatten()
        keep = np.where(targets_flat != self._ignore_label)

        loss = _logloss(targets_flat[keep], pred_flat[keep])
        loss.shape = (1,) + loss.shape
        return None, loss

    def backward(self, state, root_gradients, variables):
        # Derivative of log loss:
        #
        # TODO

        dummy = [k for k in variables]
        print("Entering backward in {} for {}".format(self.name, dummy[0]))

        if self.inputs[0] in variables:
            keep = state

            # TODO: compute proper gradients

            pred_grads = np.zeros(self.inputs[0].shape, dtype=np.float32)
            pred_grads.shape = (1,) + pred_grads.shape
            gradients = pred_grads

            variables[self.inputs[0]] = gradients

        if self.inputs[1] in variables:
            dummy_grads = np.zeros(self.inputs[1].shape, dtype=np.float32)
            dummy_grads.shape = (1,) + dummy_grads.shape

            variables[self.inputs[1]] = dummy_grads

def _logloss(act, pred):
    if len(act) == 0:
        return 0.0
    epsilon = 1e-7
    pred = sp.maximum(epsilon, pred)
    pred = sp.minimum(1-epsilon, pred)
    ll = sum(act*sp.log(pred) + sp.subtract(1,act)*sp.log(sp.subtract(1,pred)))
    ll = ll * -1.0/len(act)
    return np.array(ll, dtype=np.float32)