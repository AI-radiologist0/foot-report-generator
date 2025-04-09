# --------------------------------------------------------
# 
# Written by Jeongmin Kim(jm.kim@dankook.ac.kr)
# 
# ----------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from yacs.config import CfgNode as CN
from .model_cfg import MODEL_EXTRAS
# # If you want to save yaml file, use below code 
# from model_cfg import MODEL_EXTRAS

_C = CN()

_C.DATA_DIR = ''
_C.OUTPUT_DIR = ''
_C.LOG_DIR = ''
_C.GPUS = 0
_C.WORKERS = 4
_C.PHASE = 'train'
_C.DEVICE = "GPU"
_C.PRINT_FREQ = 1


# Cudnn related params
_C.CUDNN = CN()
_C.CUDNN.BENCHMARK = True
_C.CUDNN.DETERMINISTIC = False
_C.CUDNN.ENABLED = True

# common params for NETWORK (Feature Extractor)
_C.MODEL = CN()
_C.MODEL.NAME = 'feature_extractor'
_C.MODEL.EXTRA = MODEL_EXTRAS[_C.MODEL.NAME]
_C.MODEL.INIT_WEIGHTS = True
_C.MODEL.PRETRAINED = True

# Params for Decoder (GPT2)
_C.DECODER = CN()
_C.DECODER.NAME = 'GPT2'
_C.DECODER.EXTRA = MODEL_EXTRAS[_C.DECODER.NAME]
_C.DECODER.INIT_WEIGHTS = True
_C.DECODER.PRETRAINED = True
_C.DECODER.USE_LORA = True

# if you want to add new params for NETWORK, Init new Params below!

# DATASET related params
_C.DATASET = CN()
_C.DATASET.ROOT = ''
_C.DATASET.JSON = 'data/merge/output.json'
_C.DATASET.TRAIN_SET = 'train'
_C.DATASET.TEST_SET = 'valid'
_C.DATASET.DATA_FORMAT = 'jpg'
_C.DATASET.INCLUDE_CLASSES = ['oa', 'normal']
_C.DATASET.AUGMENT = True
_C.DATASET.BALANCE = True
_C.DATASET.BASIC_TRANSFORM = True
_C.DATASET.BBOX_CROP_FLAG = False
_C.DATASET.MEAN = [0.1147, 0.1147, 0.1147]
_C.DATASET.STD = [0.2194, 0.2194, 0.2194]
# _C.DATASET.SPLIT_RATIO = {'train': 0.7, 'validation': 0.15, 'test': 0.15}
_C.DATASET.TARGET_CLASSES = ['oa', 'normal']

_C.DATASET.PKL = 'data/pkl/output.pkl'
_C.DATASET.USE_PKL = True
_C.DATASET.USE_RAW = True
_C.DATASET.USE_PATCH = True
_C.DATASET.REPORT = False


# Related Train Parameter
_C.TRAIN = CN()
_C.TRAIN.LR_FACTOR = 0.1
_C.TRAIN.LR_STEP = [90, 110]
_C.TRAIN.LR = 0.001
_C.TRAIN.USE_GRAD_CLIP = False
_C.TRAIN.GRADIENT_ACCUMULATION_STEPS = 2

_C.TRAIN.OPTIMIZER = 'adamw'
_C.TRAIN.MOMENTUM = 0.9
_C.TRAIN.WD = 0.0001
_C.TRAIN.NESTEROV = False
_C.TRAIN.GAMMA1 = 0.99
_C.TRAIN.GAMMA2 = 0.0
_C.TRAIN.DECAY = 5e-4
_C.TRAIN.LOSS = 'BCELoss'

_C.TRAIN.SCHEDULER = 'ReduceLROnPlateau'
_C.TRAIN.MODE = 'min'
_C.TRAIN.factor = 0.5
_C.TRAIN.PATIENCE = 5

_C.TRAIN.BEGIN_EPOCH = 0
_C.TRAIN.END_EPOCH = 140

_C.TRAIN.RESUME = False
_C.TRAIN.CHECKPOINT = ''

_C.TRAIN.BATCH_SIZE_PER_GPU = 32
_C.TRAIN.SHUFFLE = True

# KFold Setting
_C.KFOLD = CN()
_C.KFOLD.USE_KFOLD = True
_C.KFOLD.KFOLD_SIZE = 5
_C.KFOLD.P = 0
_C.KFOLD.TEST_SET_RATIO = 0.15

# testing
_C.TEST = CN()

# size of images for each device
_C.TEST.BATCH_SIZE_PER_GPU = 32
_C.TEST.TEST_SET_RATIO = 0.15

# debug
_C.DEBUG = CN()
_C.DEBUG.DEBUG = False
_C.DEBUG.GRAPH_DEBUG = True


def update_config(cfg, args):
    cfg.defrost()
    cfg.merge_from_file(args.cfg)
    # cfg.merge_from_list(args.opts)

    # if args.modelDir:
    #     cfg.OUTPUT_DIR = args.modelDir
    #
    # if args.logDir:
    #     cfg.LOG_DIR = args.logDir
    #
    # if args.dataDir:
    #     cfg.DATA_DIR = args.dataDir

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
