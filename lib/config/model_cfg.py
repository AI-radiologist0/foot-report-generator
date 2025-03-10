# --------------------------------------------------------
# 
# Written by Jeongmin Kim(jm.kim@dankook.ac.kr)
# 
# ----------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from yacs.config import CfgNode as CN

# Your model related params
# examples
VGG19 = CN(new_allowed=True)
VGG19.INPUT_SIZE = [224, 224, 3]
VGG19.NUM_CLASSES = 2
VGG19.DROPOUT = 0.5

YOLOv5 = CN(new_allowed=True)
YOLOv5.CFG = './experiments/YOLO/model/yolov5.yaml'
# YOLOv5 specific configurations
YOLOv5.INPUT_SIZE = [640, 640, 3]  # Input resolution
YOLOv5.NUM_CLASSES = 1  # Number of object classes
YOLOv5.ANCHORS = [
    [10, 13, 16, 30, 33, 23],       # Small objects
    [30, 61, 62, 45, 59, 119],      # Medium objects
    [116, 90, 156, 198, 373, 326],  # Large objects
]
YOLOv5.STRIDES = [8, 16, 32]  # Stride values for each scale
YOLOv5.IOU_THRESHOLD = 0.45  # IoU threshold for NMS
YOLOv5.SCORE_THRESHOLD = 0.25  # Confidence threshold for object detection
YOLOv5.NUM_ANCHORS = 3  # Number of anchors per scale
YOLOv5.BACKBONE = "CSPDarknet53"  # Default backbone network
YOLOv5.FPN = True  # Use Feature Pyramid Network
YOLOv5.PAN = True  # Use Path Aggregation Network
YOLOv5.PT = './runs/detect/weights/best.pt'

feature_extractor = CN(new_allowed=True)
feature_extractor.RAW = 'swin-t'
feature_extractor.PATCH = 'Resnet'

MODEL_EXTRAS = {
    'VGG19': VGG19,
    'YOLOv5': YOLOv5,
    'feature_extractor': feature_extractor
}
