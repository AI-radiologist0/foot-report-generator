import os
import torch
import concurrent.futures
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm

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

# class FootPatchesDataset(Dataset):
#     def __init__(self, config, data, image_transform=train_transform, patch_transform=patch_transform):
#         self.data = data
#         self.image_transform = image_transform
#         self.patch_transform = patch_transform
        
#         self.use_raw = config.DATASET.USE_RAW
#         self.use_patches = config.DATASET.USE_PATCH
#         self.target_classes = config.DATASET.TARGET_CLASSES
#         self.num_classes = len(self.target_classes)
        
#         if not self.use_raw and self.use_patches:
#             raise AttributeError("Patches cannot be used without raw images.")
        
#         self._image_db = []
#         self._patch_db = []
#         self._labels = []
        
#         self._generate_db()
        
#         self._db = {
#             "image": self._image_db,
#             "patch": self._patch_db,
#             "label": self._labels,
#         }

#     def __len__(self):
#         return len(self._labels)

#     def _generate_db(self):
#         # balanced_data = prepare_binary_data(self.data, self.target_classes)
#         binary = True
#         if self.num_classes > 2:
#             binary = False
#         is_binary = self.num_classes == 2
#         balanced_data = prepare_data(self.data, self.target_classes, binary=is_binary)


#         # 병렬 처리를 위한 ThreadPoolExecutor 사용 (GIL 문제를 피하기 위해)
#         with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
#             results = list(tqdm(executor.map(FootPatchesDataset._process_entry, balanced_data, [self.target_classes] * len(balanced_data)), total=len(balanced_data)))

#         for image, patches, label in results:
#             self._image_db.append(image)
#             self._patch_db.append(patches)
#             self._labels.append(label)

#     @staticmethod
#     def _process_entry(entry, target_classes):
#         """ 독립적인 함수로 변환하여 multiprocessing에서 직렬화 가능하게 함 """
#         image = None
#         patches = None
#         label = 1 if entry['class'].lower() == target_classes[0] else 0

#         # Load raw image
#         if 'file_path' in entry:
#             image = np.array(Image.open(entry['file_path']).convert('RGB'))

#         # Load patches
#         if 'bbx' in entry and len(entry['bbx']) > 0:
#             patch_images = [np.array(Image.fromarray(patch)) for patch in entry['bbx'][:34]]
#         else:
#             patch_images = []

#         return image, patch_images, label

#     def __getitem__(self, idx):
#         image = self._db['image'][idx] if self.use_raw else None
#         patch_images = self._db['patch'][idx] if self.use_patches else []
#         label = self._db['label'][idx]

#         # Tensor 변환을 이곳에서 수행 (Multiprocessing에서 발생하는 문제 해결)
#         if image is not None:
#             if isinstance(image, np.ndarray):  
#                 image = Image.fromarray(image)  # 변환 추가
#             image = self.image_transform(image)  # transform 적용

#         if patch_images:
#             patch_tensors = []
#             for patch in patch_images:
#                 if isinstance(patch, np.ndarray):  
#                     patch = Image.fromarray(patch)  # 변환 추가
#                 patch_tensors.append(self.patch_transform(patch))
#             patch_tensor = torch.cat(patch_tensors, dim=0)  # Shape: (34*3, 112, 112)
#         else:
#             patch_tensor = torch.zeros(34 * 3, 112, 112) 
        
#         if self.num_classes > 2:
#             label = torch.tensor(label, dtype=torch.long) # Using Focal Loss Based CE
#         else:
#             label = torch.tensor(label, dtype=torch.float32).unsqueeze(0) # Using BCE

#         return image, patch_tensor, label

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
        self.data = balanced_data  # `self.data`에 `balanced_data` 할당

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
        report = entry.get("diagnosis", None)  # 보고서 정보 가져오기

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