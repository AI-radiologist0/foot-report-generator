# Foot Report Generator

## Overview

**Foot Report Generator** is a deep learning pipeline for foot medical image analysis. It extracts keypoints, patches, and bounding boxes from images, and supports classification/diagnosis experiments. The project leverages PyTorch, YOLO, custom models, dataset loaders, experiment automation, visualization, and wandb integration—making it suitable for both research and practical applications.

---

## Directory Structure

```
.
├── tool/                # Main scripts for training, inference, preprocessing, etc.
├── lib/                 # Core library: models, datasets, utilities
│   ├── models/          # Network architectures (ResNet, YOLO, Feature Extractor, etc.)
│   ├── dataset/         # Custom datasets and loaders
│   ├── utils/           # Visualization, evaluation, and utility functions
│   └── core/            # Core logic (e.g., Trainer)
├── config/              # Experiment/model configuration files (yaml/json)
├── output/              # Experiment results, model checkpoints, logs
├── wandb/               # wandb experiment logs
├── sampling_output/     # Intermediate outputs (sampling, logs, etc.)
├── embedding/, decoder_tool/, vector_tool/, vis_graph/, bbx_visualization/
│                        # Additional analysis/visualization/embedding tools
├── data/                # Raw data, json, pkl, images, etc.
├── environment.yml      # Conda environment and dependencies
├── README.md            # (This file)
└── ...
```

---

## Main Features

- **Data Preprocessing & Structuring**  
  - Extracts keypoints, bounding boxes, and patches from images and stores them in pickle files.
- **Deep Learning-based Classification/Diagnosis**  
  - Supports various models (feature_extractor, resnet, yolo, etc.)
  - Config-driven experiment automation and wandb integration.
- **Visualization & Analysis**  
  - ROC curve, tensorboard, wandb, bbox/keypoint visualization, and more.
- **Utilities**  
  - EarlyStopping, BestModelSaver, stratified split, and other research-friendly tools.

---

## Joint Detector

The **joint detector** is a key module for detecting anatomical keypoints (joints) in foot images. It is implemented in [`tool/joint_detector.py`](tool/joint_detector.py) and is based on a deep convolutional neural network (typically a ResNet backbone with deconvolution layers, e.g., `pose_resnet`).

### Main Features

- **Input:** Foot images (e.g., X-ray or photographic images).
- **Output:** 2D coordinates of anatomical keypoints (joints) for each foot in the image.
- **Model:** Uses a configurable pose estimation network (default: `pose_resnet`), which outputs heatmaps for each joint.
- **Configurable:** All settings (model, dataset, augmentation, etc.) are controlled via YAML config files (see `config/debugging_for_joint_detector_config.yaml` for an example).

### How to Use

You can run the joint detector as a standalone script:

```bash
python tool/joint_detector.py --cfg config/debugging_for_joint_detector_config.yaml
```

- The script loads the model, runs inference on the test set, and outputs detected keypoints.
- Supports options for using detected or ground-truth bounding boxes, flipping, post-processing, and more (see script arguments).

### Model Architecture

- The default model is a ResNet-based pose estimation network (`pose_resnet`), which is widely used for human/animal keypoint detection.
- The model outputs a set of heatmaps, one per joint, from which the 2D coordinates are extracted.

### Configuration Example

```yaml
MODEL:
  NAME: 'pose_resnet'
  PRETRAINED: 'ckpt/detector/final_state.pth.tar'
  IMAGE_SIZE: [288, 384]
  NUM_JOINTS: 17
  EXTRA:
    HEATMAP_SIZE: [72, 96]
    SIGMA: 2
    NUM_DECONV_LAYERS: 3
    NUM_DECONV_FILTERS: [256, 256, 256]
    ...
DATASET:
  DATASET: 'coco'
  ROOT: 'data/coco/'
  TEST_SET: 'JPEGImages'
  ...
```

### Output

- The detected keypoints can be used for downstream tasks such as patch extraction, classification, or visualization.

---

## Getting Started

### 1. Environment Setup

```bash
conda env create -f environment.yml
conda activate biobert_lora_env
```

### 2. Data Preparation

- Place your data files (json, pkl, images, etc.) in the `data/` directory.
- Adjust data paths in your config file (e.g., `config/large/tmp/origin_oa_normal.yaml`).

### 3. Running Experiments

Example: Run 20 repeated experiments with a specific config file

```bash
python tool/patch_train_exp.py --cfg config/large/tmp/origin_oa_normal.yaml --repeat 20
```

- Results will be saved in the `output/` and `wandb/` directories.

### 4. Checking Results

- Use tensorboard, wandb, and files in the output folder for analysis and visualization.

---

## Configuration

Config files are in YAML format. Example key sections:

```yaml
MODEL:
  NAME: feature_extractor2
  PRETRAINED: True
  EXTRA:
    WITH_ATTN: False
    ONLYCAT: True
    VIEWCAT: False
  FREEZE:
    BACKBONE: True
    PROJECTION: False
    CLASSIFIER: False
DATASET:
  INCLUDE_CLASSES: ['oa', 'normal']
  JSON: data/json/tmp0418/joint/final_samples_both_only_v2.json
  PKL: data/pkl/output200x300.pkl
  TARGET_CLASSES: ['oa', 'normal']
TRAIN:
  BATCH_SIZE_PER_GPU: 48
  BEGIN_EPOCH: 0
  END_EPOCH: 50
  OPTIMIZER: adam
  LR: 0.001
  ...
```

---

## Pickle File Structure

Each pickle file is a Python dictionary where:

- Each **image index (integer)** is a key.
- The value is a nested dictionary containing metadata and extracted data.

Example structure:

```
{
    0: {
        "meta": {
            "file_path": "path/to/image.jpg",
            "patient_id": "CAUHRA20004",
            "diagnosis": "hallux valgus, Rt ...",
            "class": "Uncertain"
        },
        "data": {
            "left_bbox": [x1, y1, x2, y2],
            "right_bbox": [x1, y1, x2, y2],
            "left_keypoints": [[x1, y1], ..., [x17, y17]],
            "right_keypoints": [[x1, y1], ..., [x17, y17]],
            "left_patches": { 1: [x1, y1, x2, y2], ... },
            "right_patches": { 1: [x1, y1, x2, y2], ... }
        }
    },
    1: { ... }
}
```

- **meta**: Metadata about the image and patient.
- **data**: Processed outputs (bounding boxes, keypoints, patches, etc.)

See the original README or code for more details and usage examples.

---

## Main Dependencies

- Python 3.8+
- PyTorch, torchvision, torchaudio
- wandb, opencv, matplotlib, seaborn, and more
- See `environment.yml` for the full list

---

## Notes

- The project includes experiment automation, model saving/loading, ROC/PR curve visualization, wandb integration, and more.
- Utilities and analysis/visualization tools are available in `tool/`, `lib/`, `bbx_visualization/`, `vis_graph/`, etc.

---

## License

See the LICENSE file for details.

---

**If you need more detailed usage examples, data structure explanations, or experiment result interpretation, feel free to ask!**

