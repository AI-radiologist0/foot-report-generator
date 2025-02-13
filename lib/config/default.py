# --------------------------------------------------------
# Reference from HRNet-Human-Pose-Estimation
# refact code from old one.
# Written by JeongMin Kim(jm.kim@dankook.ac.kr)
# ----------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from yacs.config import CfgNode as CN
# from .model import MODEL_EXTRAS
from .model_cfg import MODEL_EXTRAS

_C = CN()

_C.DATA_DIR = ''
_C.OUTPUT_DIR = ''
_C.LOG_DIR = ''
_C.GPUS = 0
_C.WORKERS = 4
_C.PHASE = 'train'

# Cudnn related params
_C.CUDNN = CN()
_C.CUDNN.BENCHMARK = True
_C.CUDNN.DETERMINISTIC = False
_C.CUDNN.ENABLED = True

# common params for NETWORK
_C.MODEL = CN()
_C.MODEL.NAME = 'FCN'
_C.MODEL.EXTRA = MODEL_EXTRAS[_C.MODEL.NAME]
_C.MODEL.INIT_WEIGHTS = True
_C.MODEL.PRETRAINED = ''

# if you want to add new params for NETWORK, Init new Params below!

# DATASET related params
_C.DATASET = CN()
_C.DATASET.ROOT = ''
_C.DATASET.TRAIN_SET = 'train'
_C.DATASET.TEST_SET = 'valid'
_C.DATASET.DATA_FORMAT = 'jpg'

# training data augmentation
# Implement your data augmentation

# train
_C.TRAIN = CN()

_C.TRAIN.LR_FACTOR = 0.1
_C.TRAIN.LR_STEP = [90, 110]
_C.TRAIN.LR = 0.001

_C.TRAIN.OPTIMIZER = 'adam'
_C.TRAIN.MOMENTUM = 0.9
_C.TRAIN.WD = 0.0001
_C.TRAIN.NESTEROV = False
_C.TRAIN.GAMMA1 = 0.99
_C.TRAIN.GAMMA2 = 0.0

_C.TRAIN.BEGIN_EPOCH = 0
_C.TRAIN.END_EPOCH = 140

_C.TRAIN.RESUME = False
_C.TRAIN.CHECKPOINT = ''

_C.TRAIN.BATCH_SIZE_PER_GPU = 32
_C.TRAIN.SHUFFLE = True

# testing
_C.TEST = CN()

# size of images for each device
_C.TEST.BATCH_SIZE_PER_GPU = 32

# debug
_C.DEBUG = CN()
_C.DEBUG.DEBUG = False


def update_config(cfg, args):
    cfg.defrost()
    cfg.merge_from_file(args.cfg)
    cfg.merge_from_list(args.opts)

    if args.modelDir:
        cfg.OUTPUT_DIR = args.modelDir

    if args.logDir:
        cfg.LOG_DIR = args.logDir

    if args.dataDir:
        cfg.DATA_DIR = args.dataDir

    # cfg.DATASET.ROOT = os.path.join(
    #     cfg.DATA_DIR, cfg.DATASET.ROOT
    # )
    #
    # cfg.MODEL.PRETRAINED = os.path.join(
    #     cfg.DATA_DIR, cfg.MODEL.PRETRAINED
    # )

    # if cfg.TEST.MODEL_FILE:
    #     cfg.TEST.MODEL_FILE = os.path.join(
    #         cfg.DATA_DIR, cfg.TEST.MODEL_FILE
    #     )

    cfg.freeze()


if __name__ == '__main__':
    import sys

    with open(sys.argv[1], 'w') as f:
        print(_C, file=f)
