# --------------------------------------------------------
# Copyright (c) 2017 Microsoft
# --------------------------------------------------------

from cntk import output_variable
from cntk.ops.functions import UserFunction
import numpy as np
from fast_rcnn.config import cfg

DEBUG = True

class SmoothL1Loss(UserFunction):
    """
    Computes a smooth L1 loss
    """

    def __init__(self, arg1, arg2, name='SmoothL1Loss', sigma=None):
        super(SmoothL1Loss, self).__init__([arg1, arg2], name=name)

        self._sigma = sigma
        self.williscrap = None

    def __del__(self):
        import pdb; pdb.set_trace()
        print("destructor")

    def infer_outputs(self):
        return [output_variable(self.inputs[0].shape, self.inputs[0].dtype, self.inputs[0].dynamic_axes)]

    def forward(self, arguments, device=None, outputs_to_retain=None):
        bottom = arguments

        # Algorithm:
        #
        # (According to Fast R-CNN paper, formula (3))
        # The smooth L1 loss is defined per dimension as
        #
        # smooth_L1(x) = | 0.5 * x^2     , if |x| < 1
        #                | |x| - 0.5     , otherwise

        predictions = bottom[0][0,:]
        targets = bottom[1][0,:]
        sigma = self._sigma

        #import pdb; pdb.set_trace()

        diff = predictions - targets
        x = np.abs(diff)
        lt1 = np.where(x < 1)
        loss = x - .5
        l2 = x * x * .5
        loss[lt1] = l2[lt1]

        loss.shape = (1,) + loss.shape
        return diff, loss

    def backward(self, state, root_gradients, variables):
        # Derivative of smooth L1 loss:
        #
        # - root_gradients      , if diff <= -1
        # diff * root_gradients , if -1 < diff < 1
        # root_gradients        , else

        dummy = [k for k in variables]
        print("Entering backward in {} for {}".format(self.name, dummy[0]))

        #import pdb; pdb.set_trace()

        if self.inputs[0] in variables:
            if False:
                dummy_grads = np.zeros(self.inputs[0].shape, dtype=np.float32)
                dummy_grads.shape = (1,) + dummy_grads.shape
                variables[self.inputs[0]] = dummy_grads
                return

            diff = state
            item_gradients = root_gradients[0,:]

            assert(item_gradients.size == diff.size)
            diff = diff.reshape(item_gradients.shape)

            le_minus_one = np.where(diff <= -1)
            ge_plus_one = np.where(diff >= 1)

            gradients = item_gradients * diff
            gradients[le_minus_one] = -1 * item_gradients[le_minus_one]
            gradients[ge_plus_one] = item_gradients [ge_plus_one]
            gradients.shape = (1,) + gradients.shape
            variables[self.inputs[0]] = gradients

            #import pdb; pdb.set_trace()
            #self.williscrap = [variables, state, root_gradients, gradients]

        if self.inputs[1] in variables:
            dummy_grads = np.zeros(self.inputs[1].shape, dtype=np.float32)
            dummy_grads.shape = (1,) + dummy_grads.shape
            variables[self.inputs[1]] = dummy_grads

        import pdb; pdb.set_trace()
