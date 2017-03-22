# --------------------------------------------------------
# Copyright (c) 2017 Microsoft
# --------------------------------------------------------

from cntk import output_variable
from cntk.ops.functions import UserFunction
import numpy as np
from fast_rcnn.config import cfg

class CntkId(UserFunction):

    def __init__(self, arg1, name='CntkId'):
        super(CntkId, self).__init__([arg1], name=name)

    def infer_outputs(self):
        return [output_variable(self.inputs[0].shape, self.inputs[0].dtype, self.inputs[0].dynamic_axes)]

    def forward(self, arguments, device=None, outputs_to_retain=None):
        #import pdb; pdb.set_trace()
        result = np.array(arguments[0])
        result.shape = (1,) + result.shape
        return None, result

    def backward(self, state, root_gradients):
        #import pdb; pdb.set_trace()
        return root_gradients

