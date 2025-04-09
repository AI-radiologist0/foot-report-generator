import os
import json, re
import torch
import concurrent.futures
from PIL import Image
import cv2
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm
from pycocomedical import COCOMedical


from utils.utils import prepare_binary_data, prepare_data

# Image transformations (전역 변수로 변경)
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

patch_transform = transforms.Compose([
    transforms.Resize((112, 112)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

class FootPatchesDataset(Dataset):
    def __init__(self, config, data, image_transform=train_transform, patch_transform=patch_transform):
        """
        - Lazy Loading을 적용한 FootPatchesDataset
        - Patch Tensor 유지
        - Binary Classification (BCE) & Multi-Class Classification (Focal Loss) 자동 지원

        Args:
            config: 환경 설정 파일
            data: 원본 데이터셋 (pkl 파일에서 로드)
            image_transform: 이미지 변환 함수
            patch_transform: 패치 변환 함수
        """
        self.data = data
        self.image_transform = image_transform
        self.patch_transform = patch_transform

        self.use_raw = config.DATASET.USE_RAW
        self.use_patches = config.DATASET.USE_PATCH
        self.target_classes = config.DATASET.TARGET_CLASSES
        self.abnormal_classify =  True if len(self.target_classes) == 2 and 'abnormal' in self.target_classes and 'normal' in self.target_classes else False
        self.abnormal_mapping = {'ra' : 'abnormal', 'oa': 'abnormal', 'gout': 'abnormal', 'normal': 'normal'} if self.abnormal_classify else None
        self.use_report = config.DATASET.REPORT

        if isinstance(self.target_classes, str):
            self.target_classes = self.target_classes.split(",")

        if not isinstance(self.target_classes, list):
            raise TypeError(f"Expected list for target_classes, but got {type(self.target_classes)}")

        self.num_classes = len(self.target_classes)
        self.is_binary = self.num_classes == 2  

        if not self.use_raw and self.use_patches:
            raise AttributeError("Patches cannot be used without raw images.")

        # According to augment and balance flag, Generate data.
        balanced_data, _, _ = prepare_data(self.data, self.target_classes, config, self.is_binary)
        self.data = balanced_data  # self.data에 balanced_data 할당

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        필요할 때만 데이터를 로드하는 Lazy Loading 방식 적용
        - 원본 이미지
        - 패치 이미지 (최대 34개, 부족하면 Zero Padding)
        - 레이블 (이진 분류 & 다중 분류 자동 적용)
        """
        entry = self.data[idx]
        file_path = entry['file_path']            
        label = self.target_classes.index(self.abnormal_mapping[entry['class'].lower()]) if self.abnormal_mapping else self.target_classes.index(entry['class'].lower()) # 정수형 라벨 변환
        patches = entry.get("bbx", [])  # 패치 정보 가져오기
        report = entry.get("diagnosis", None)  # 보고서 정보 

        # **Lazy Load: 원본 이미지 로드**
        image = Image.open(file_path).convert("RGB")
        image = self.image_transform(image)

        # **Lazy Load: Patch 이미지 로드 및 변환**
        patch_tensors = []
        for patch in patches[:34]:  # 최대 34개의 패치 사용
            patch_img = Image.fromarray(np.array(patch))
            patch_tensors.append(self.patch_transform(patch_img))

        # **패치 개수가 34개보다 적을 경우 Zero Padding 추가**
        num_patches = len(patch_tensors)
        if num_patches < 34:
            padding = [torch.zeros_like(patch_tensors[0])] * (34 - num_patches)
            patch_tensors.extend(padding)

        # **패치 텐서 병합 (최종 Shape: (34*3, 112, 112))**
        patch_tensor = torch.cat(patch_tensors, dim=0) if patch_tensors else torch.zeros(34 * 3, 112, 112)

        # **Binary vs Multi-class 레이블 변환**
        if self.is_binary:
            label = torch.tensor(label, dtype=torch.float32).unsqueeze(0)  # BCE Loss
        else:
            label = torch.tensor(label, dtype=torch.long)  # CrossEntropy Loss 기반 Focal Loss

        if self.use_report:
            return image, patch_tensor, label, report
        return image, patch_tensor, label

class FootPatchesDatasetWithJson(Dataset):
    def __init__(self, cfg, image_transform=train_transform, patch_transform=patch_transform):
        """
        JSON 기반 Lazy Loading FootPatchesDataset

        Args:
            cfg: 설정 파일 (json 경로 포함)
            image_transform: 이미지 변환 함수
            patch_transform: 패치 변환 함수
        """
        self.cfg = cfg
        self.image_transform = image_transform
        self.patch_transform = patch_transform

        self.use_raw = cfg.DATASET.USE_RAW
        self.use_patches = cfg.DATASET.USE_PATCH
        self.target_classes = cfg.DATASET.TARGET_CLASSES
        self.abnormal_classify = (
            len(self.target_classes) == 2 and 'abnormal' in self.target_classes and 'normal' in self.target_classes
        )
        self.abnormal_mapping = {'ra': 'abnormal', 'oa': 'abnormal', 'gout': 'abnormal', 'normal': 'normal'} \
            if self.abnormal_classify else None
        self.use_report = cfg.DATASET.REPORT

        if isinstance(self.target_classes, str):
            self.target_classes = self.target_classes.split(",")

        if not isinstance(self.target_classes, list):
            raise TypeError(f"Expected list for target_classes, but got {type(self.target_classes)}")

        self.num_classes = len(self.target_classes)
        self.is_binary = self.num_classes == 2

        if not self.use_raw and self.use_patches:
            raise AttributeError("Patches cannot be used without raw images.")

        # disease_list 구성
        coco_medical = COCOMedical()
        coco_medical.load_json(cfg.DATASET.JSON)

        self.data = {}
        for idx, value in enumerate(coco_medical.diseases):
            # self.data.append(coco_medical.diseases[value].to_dict())
            self.data[idx] = coco_medical.diseases[value].to_dict()
        
        if self.is_binary:
            balanced_data, _, _ = prepare_data(self.data, self.target_classes, cfg, self.is_binary)
            self.data = balanced_data
        else:
            self.data = self.data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        entry = self.data[idx]
        file_path = entry['file_path']
        label_str = self.abnormal_mapping[entry['class_label'].lower()] \
            if self.abnormal_mapping else entry['class_label'].lower()
        label = self.target_classes.index(label_str)
        report = self._clean_report(entry.get("diagnosis", ""))

        # Load image
        image = Image.open(file_path).convert("RGB")
        image = self.image_transform(image)

        # Generate patches
        patches = self.generate_patches_from_keypoints(file_path, entry['keypoint_id'])

        # Transform patches
        patch_tensors = [self.patch_transform(Image.fromarray(p)) for p in patches]
        patch_tensor = torch.cat(patch_tensors, dim=0)

        if self.is_binary:
            label = torch.tensor(label, dtype=torch.float32).unsqueeze(0)
        else:
            label = torch.tensor(label, dtype=torch.long)

        if self.use_report:
            return image, patch_tensor, label, report
        return image, patch_tensor, label

    def generate_patches_from_keypoints(self, image_path, keypoints_dict, crop_size=(200, 300), patch_size=(224, 224)):
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Cannot read image at {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        def extract(keypoints_side):
            patches = []
            keypoints = keypoints_side[0]['keypoints']
            for i in range(17):
                x, y, score = int(keypoints[i * 3]), int(keypoints[i * 3 + 1]), keypoints[i * 3 + 2]
                if score > 0.0:
                    x_min = max(x - crop_size[0] // 2, 0)
                    y_min = max(y - crop_size[1] // 2, 0)
                    x_max = min(x + crop_size[0] // 2, image.shape[1])
                    y_max = min(y + crop_size[1] // 2, image.shape[0])
                    crop = image[y_min:y_max, x_min:x_max]
                    if crop.size > 0:
                        try:
                            resized = cv2.resize(crop, patch_size)
                            patches.append(resized)
                        except:
                            continue
            return patches

        left_patches, right_patches = [], []
        if 'left' in keypoints_dict:
            if keypoints_dict['left']:
                left_patches = extract(keypoints_dict['left'])
        
        if 'right' in keypoints_dict:
            if keypoints_dict['right']:
                right_patches = extract(keypoints_dict['right'])

        if left_patches and not right_patches:
            right_patches = [cv2.flip(p, 1) for p in left_patches]
        elif right_patches and not left_patches:
            left_patches = [cv2.flip(p, 1) for p in right_patches]
        elif not left_patches and not right_patches:
            black = np.zeros((patch_size[1], patch_size[0], 3), dtype=np.uint8)
            return [black] * 34

        def pad(p):
            black = np.zeros((patch_size[1], patch_size[0], 3), dtype=np.uint8)
            while len(p) < 17:
                p.append(black)
            return p[:17]

        left_patches = pad(left_patches)
        right_patches = pad(right_patches)
        return left_patches + right_patches

    def _clean_report(self, report):
        if not report:
            return ""
        report = re.sub(r"_x000D_", "\n", report)
        report = report.replace("\r", "").strip()
        return report
