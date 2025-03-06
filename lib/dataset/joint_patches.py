import os
import torch
import concurrent.futures
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm

from utils.utils import prepare_binary_data

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
        self.data = data
        self.image_transform = image_transform
        self.patch_transform = patch_transform
        
        self.use_raw = config.DATASET.USE_RAW
        self.use_patches = config.DATASET.USE_PATCH
        self.target_classes = config.DATASET.TARGET_CLASSES
        
        if not self.use_raw and self.use_patches:
            raise AttributeError("Patches cannot be used without raw images.")
        
        self._image_db = []
        self._patch_db = []
        self._labels = []
        
        self._generate_db()
        
        self._db = {
            "image": self._image_db,
            "patch": self._patch_db,
            "label": self._labels,
        }

    def __len__(self):
        return len(self._labels)

    def _generate_db(self):
        balanced_data = prepare_binary_data(self.data, self.target_classes)

        # 병렬 처리를 위한 ThreadPoolExecutor 사용 (GIL 문제를 피하기 위해)
        with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
            results = list(tqdm(executor.map(FootPatchesDataset._process_entry, balanced_data, [self.target_classes] * len(balanced_data)), total=len(balanced_data)))

        for image, patches, label in results:
            self._image_db.append(image)
            self._patch_db.append(patches)
            self._labels.append(label)

    @staticmethod
    def _process_entry(entry, target_classes):
        """ 독립적인 함수로 변환하여 multiprocessing에서 직렬화 가능하게 함 """
        image = None
        patches = None
        label = 1 if entry['class'].lower() == target_classes[0] else 0

        # Load raw image
        if 'file_path' in entry:
            image = np.array(Image.open(entry['file_path']).convert('RGB'))

        # Load patches
        if 'bbx' in entry and len(entry['bbx']) > 0:
            patch_images = [np.array(Image.fromarray(patch)) for patch in entry['bbx'][:34]]
        else:
            patch_images = []

        return image, patch_images, label

    def __getitem__(self, idx):
        image = self._db['image'][idx] if self.use_raw else None
        patch_images = self._db['patch'][idx] if self.use_patches else []

        # Tensor 변환을 이곳에서 수행 (Multiprocessing에서 발생하는 문제 해결)
        if image is not None:
            if isinstance(image, np.ndarray):  
                image = Image.fromarray(image)  # 변환 추가
            image = self.image_transform(image)  # transform 적용

        if patch_images:
            patch_tensors = []
            for patch in patch_images:
                if isinstance(patch, np.ndarray):  
                    patch = Image.fromarray(patch)  # 변환 추가
                patch_tensors.append(self.patch_transform(patch))
            patch_tensor = torch.cat(patch_tensors, dim=0)  # Shape: (34*3, 112, 112)
        else:
            patch_tensor = torch.zeros(34 * 3, 112, 112) 

        label = self._db['label'][idx]
        return image, patch_tensor, label
