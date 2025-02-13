# --------------------------------------------------------
# Reference from HRNet-Human-Pose-Estimation
# refact code from old one.
# Written by JeongMin Kim(jm.kim@dankook.ac.kr)
# ----------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from yacs.config import CfgNode as CN

# Your model related params
# examples
FCN = CN(new_allowed=True)
FCN.INPUT_CHANNELS = 784
FCN.HIDDEN_CHANNELS = [128, 64, 10]
FCN.HIDDEN_ACTIVATION = 'ReLU'
FCN.HIDDEN_DROPOUT = 0.25
FCN.OUTPUT_CHANNELS = 2
FCN.OUTPUT_ACTIVATION = 'logSoftMax'

MODEL_EXTRAS = {
    'FCN': FCN,
}
