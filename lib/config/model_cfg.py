from yacs.config import CfgNode as CN

# Your model related params
# Existing configurations (VGG19, YOLOv5, etc.)
VGG19 = CN(new_allowed=True)
VGG19.INPUT_SIZE = [224, 224, 3]
VGG19.NUM_CLASSES = 2
VGG19.DROPOUT = 0.5

YOLOv5 = CN(new_allowed=True)
YOLOv5.CFG = './experiments/YOLO/model/yolov5.yaml'
YOLOv5.INPUT_SIZE = [640, 640, 3]
YOLOv5.NUM_CLASSES = 1
YOLOv5.ANCHORS = [
    [10, 13, 16, 30, 33, 23],
    [30, 61, 62, 45, 59, 119],
    [116, 90, 156, 198, 373, 326],
]
YOLOv5.STRIDES = [8, 16, 32]
YOLOv5.IOU_THRESHOLD = 0.45
YOLOv5.SCORE_THRESHOLD = 0.25
YOLOv5.NUM_ANCHORS = 3
YOLOv5.BACKBONE = "CSPDarknet53"
YOLOv5.FPN = True
YOLOv5.PAN = True
YOLOv5.PT = './runs/detect/weights/best.pt'

feature_extractor = CN(new_allowed=True)
feature_extractor.RAW = 'swin-t'
feature_extractor.PATCH = 'Resnet'
feature_extractor.CKPT = 'wandb/run-multiclass-classifier/files/best_model.pth'
feature_extractor.USE_CKPT = False
feature_extractor.FREEZE = True
feature_extractor.CLASSIFIER_HEAD = False

feature_extractor2 = CN(new_allowed=True)
feature_extractor2.RAW = 'swin-t'
feature_extractor2.PATCH = 'Resnet'
feature_extractor2.CKPT = 'wandb/run-ra-binary-classifier/files/best_model.pth'
feature_extractor2.USE_CKPT = False
feature_extractor2.FREEZE = True
feature_extractor2.CLASSIFIER_HEAD = False
feature_extractor2.WITH_ATTN = True
feature_extractor2.ONLYCAT = False
feature_extractor2.VIEWCAT = False




# Add GPT-2 Configuration
GPT2 = CN(new_allowed=True)
GPT2.MODEL_SIZE = 'small'  # Options: 'small', 'medium'
GPT2.MAX_SEQ_LENGTH = 512  # Maximum sequence length for the model
GPT2.TRAIN_BATCH_SIZE = 16  # Training batch size
GPT2.VAL_BATCH_SIZE = 8    # Validation batch size
GPT2.TEST_BATCH_SIZE = 8   # Testing batch size
GPT2.LEARNING_RATE = 5e-5  # Learning rate for training
GPT2.OPTIMIZER = 'AdamW'   # Optimizer to use
GPT2.PT = 'wandb/run-20250313_153146-anjnooj3/files/best_model.pth'

# Add GPT-2 to MODEL_EXTRAS
MODEL_EXTRAS = {
    'VGG19': VGG19,
    'YOLOv5': YOLOv5,
    'feature_extractor': feature_extractor,
    'feature_extractor2': feature_extractor2,
    'GPT2': GPT2,  # New GPT-2 configuration
}
