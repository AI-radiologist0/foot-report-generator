import os
import logging
import json, re
import unicodedata
import torch
import concurrent.futures
from PIL import Image
import cv2
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm
from pycocomedical import COCOMedical
import torch.nn.functional as F
import albumentations as A
from albumentations.pytorch import ToTensorV2


from utils.utils import prepare_binary_data, prepare_data, prepare_data_with_seed

# 3채널용 transform (원본 이미지)
image_transform_rgb = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.1147, 0.1147, 0.1147], std=[0.2194, 0.2194, 0.2194])
])

# 1채널용 transform (패치)
def get_patch_transform_gray(concat_patch):
    size = (28, 28) if concat_patch else (112, 112)
    return transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.1147], std=[0.2194])
    ])

def check_image_channels(image_path):
    """
    이미지의 각 채널값이 동일한지 확인하는 유틸리티 함수
    
    Args:
        image_path: 이미지 파일 경로
        
    Returns:
        dict: 채널 정보를 담은 딕셔너리
    """
    try:
        img = Image.open(image_path)
        original_mode = img.mode
        original_size = img.size
        
        # RGB로 변환
        img_rgb = img.convert("RGB")
        rgb_array = np.array(img_rgb)
        
        # 각 채널 추출
        r_channel = rgb_array[:, :, 0]
        g_channel = rgb_array[:, :, 1]
        b_channel = rgb_array[:, :, 2]
        
        # 채널 간 차이 계산
        r_g_diff = np.abs(r_channel - g_channel)
        r_b_diff = np.abs(r_channel - b_channel)
        g_b_diff = np.abs(g_channel - b_channel)
        
        # Grayscale 여부 판단 (완전히 동일한 경우만)
        is_grayscale = np.max(r_g_diff) == 0 and np.max(r_b_diff) == 0 and np.max(g_b_diff) == 0
        
        return {
            'file_path': image_path,
            'original_mode': original_mode,
            'original_size': original_size,
            'rgb_shape': rgb_array.shape,
            'r_g_max_diff': np.max(r_g_diff),
            'r_b_max_diff': np.max(r_b_diff),
            'g_b_max_diff': np.max(g_b_diff),
            'r_g_mean_diff': np.mean(r_g_diff),
            'r_b_mean_diff': np.mean(r_b_diff),
            'g_b_mean_diff': np.mean(g_b_diff),
            'is_grayscale': is_grayscale,
            'r_mean': np.mean(r_channel),
            'g_mean': np.mean(g_channel),
            'b_mean': np.mean(b_channel)
        }
        
    except Exception as e:
        return {
            'file_path': image_path,
            'error': str(e)
        }

class FootPatchesDataset(Dataset):
    def __init__(self, config, data, image_transform=image_transform_rgb, patch_transform=None):
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
        self.image_transform = image_transform  # 3채널용
        self.patch_transform = patch_transform if patch_transform is not None else get_patch_transform_gray(config.DATASET.CONCAT_PATCH) # 1채널용

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

    def analyze_channels(self, max_samples=50):
        """
        데이터셋의 이미지들을 분석하여 채널 정보를 확인
        
        Args:
            max_samples: 분석할 최대 샘플 수
            
        Returns:
            dict: 분석 결과
        """
        print(f"데이터셋 채널 분석 시작 (최대 {max_samples}개 샘플)")
        
        results = []
        grayscale_count = 0
        non_grayscale_count = 0
        
        for i, entry in enumerate(tqdm(self.data[:max_samples], total=min(max_samples, len(self.data)))):
            file_path = entry['file_path']
            result = check_image_channels(file_path)
            results.append(result)
            
            if 'error' not in result:
                if result['is_grayscale']:
                    grayscale_count += 1
                else:
                    non_grayscale_count += 1
        
        # 결과 분석
        print(f"\n=== 채널 분석 결과 ===")
        print(f"총 분석된 이미지: {len(results)}")
        print(f"Grayscale 이미지: {grayscale_count}")
        print(f"Non-grayscale 이미지: {non_grayscale_count}")
        print(f"Grayscale 비율: {grayscale_count/len(results)*100:.2f}%")
        
        # Non-grayscale 이미지들의 상세 정보
        non_grayscale_results = [r for r in results if 'error' not in r and not r['is_grayscale']]
        
        if non_grayscale_results:
            print(f"\n=== Non-grayscale 이미지 상세 정보 (처음 5개) ===")
            for i, result in enumerate(non_grayscale_results[:5]):
                print(f"\n{i+1}. {os.path.basename(result['file_path'])}")
                print(f"   원본 모드: {result['original_mode']}")
                print(f"   R-G 최대 차이: {result['r_g_max_diff']}")
                print(f"   R-B 최대 차이: {result['r_b_max_diff']}")
                print(f"   G-B 최대 차이: {result['g_b_max_diff']}")
                print(f"   R 평균: {result['r_mean']:.2f}")
                print(f"   G 평균: {result['g_mean']:.2f}")
                print(f"   B 평균: {result['b_mean']:.2f}")
        
        # 에러가 있는 이미지들
        error_results = [r for r in results if 'error' in r]
        if error_results:
            print(f"\n=== 에러가 발생한 이미지들 ===")
            for result in error_results:
                print(f"  {os.path.basename(result['file_path'])}: {result['error']}")
        
        return {
            'total_analyzed': len(results),
            'grayscale_count': grayscale_count,
            'non_grayscale_count': non_grayscale_count,
            'grayscale_ratio': grayscale_count/len(results)*100 if len(results) > 0 else 0,
            'results': results
        }

    def __getitem__(self, idx):
        """
        필요할 때만 데이터를 로드하는 Lazy Loading 방식 적용
        - 원본 이미지
        - 패치 이미지 (최대 34개, 부족하면 Zero Padding)
        - 레이블 (이진 분류 & 다중 분류 자동 적용)
        """

        meta = {}

        entry = self.data[idx]
        file_path = entry['file_path']
        label = self.target_classes.index(self.abnormal_mapping[entry['class'].lower()]) if self.abnormal_mapping else self.target_classes.index(entry['class'].lower()) # 정수형 라벨 변환
        patches = entry.get("bbx", [])  # 패치 정보 가져오기
        report = entry.get("diagnosis", None)  # 보고서 정보 

        meta['original_label'] = entry['class'].lower()
        meta['label'] = label
        meta['patient_id'] = entry['patient_id']

        # **원본 이미지는 3채널로 로딩**
        image = Image.open(file_path).convert("RGB")
        image = self.image_transform(image)

        # **패치는 1채널로 로딩**
        patch_tensors = []
        for patch in patches[:34]:  # 최대 34개의 패치 사용
            patch_img = Image.fromarray(np.array(patch)).convert("L")  # 1채널로 변환
            patch_tensor = self.patch_transform(patch_img)
            patch_tensors.append(patch_tensor)
        # **패치 개수가 34개보다 적을 경우 Zero Padding 추가**
        num_patches = len(patch_tensors)
        if num_patches < 34:
            if num_patches > 0:
                padding = [torch.zeros_like(patch_tensors[0])] * (34 - num_patches)
            else:
                padding = [torch.zeros(1, 112, 112)] * 34
            patch_tensors.extend(padding)
        # **패치 텐서 병합 (최종 Shape: (34, 1, 112, 112))**
        if patch_tensors:
            patch_tensor = torch.stack(patch_tensors, dim=0)
        else:
            patch_tensor = torch.zeros(34, 1, 112, 112)
        # patch_tensor = torch.cat(patch_tensors, dim=0) if patch_tensors else torch.zeros(34 * 3, 112, 112)

        # **Binary vs Multi-class 레이블 변환**
        if self.is_binary:
            label = torch.tensor(label, dtype=torch.float32).unsqueeze(0)  # BCE Loss
        else:
            label = torch.tensor(label, dtype=torch.long)  # CrossEntropy Loss 기반 Focal Loss

        if self.use_report:
            return image, patch_tensor, label, report
        return image, patch_tensor, label, meta

class FootPatchesDatasetWithJson(Dataset):
    eos_token = '<eos>'
    def __init__(self, cfg, image_transform=image_transform_rgb, patch_transform=None):
        """
        JSON 기반 Lazy Loading FootPatchesDataset

        Args:
            cfg: 설정 파일 (json 경로 포함)
            image_transform: 이미지 변환 함수
            patch_transform: 패치 변환 함수
        """
        self.cfg = cfg
        self.image_transform = image_transform  # 3채널용
        self.patch_transform = patch_transform if patch_transform is not None else get_patch_transform_gray(cfg.DATASET.CONCAT_PATCH) # 1채널용

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
        patch_tensors = []
        for p in patches:
            patch_tensor = self.patch_transform(Image.fromarray(p).convert("L"))
            patch_tensors.append(patch_tensor)
        if len(patch_tensors) < 34:
            if len(patch_tensors) > 0:
                padding = [torch.zeros_like(patch_tensors[0])] * (34 - len(patch_tensors))
            else:
                padding = [torch.zeros(1, 112, 112)] * 34
            patch_tensors.extend(padding)
        if patch_tensors:
            patch_tensor = torch.stack(patch_tensors, dim=0)
        else:
            patch_tensor = torch.zeros(34, 1, 112, 112)
        # patch_tensor = torch.cat(patch_tensors, dim=0)

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

    def _clean_report(self, text):
        # Normalize and remove non-ASCII characters.
        text = unicodedata.normalize('NFKC', text)
        text = re.sub(r'[^\x00-\x7F]+', ' ', text)
        text = re.sub(r'([.!?]){2,}', r'\1', text)
        text = re.sub(r'\[\s*finding\s*\]', '[FINDING]', text, flags=re.IGNORECASE)
        text = re.sub(r'\[\s*conclusion\s*\]', '[CONCLUSION]', text, flags=re.IGNORECASE)
        text = re.sub(r'\[\s*diagnosis\s*\]', '[DIAGNOSIS]', text, flags=re.IGNORECASE)
        parts = re.split(r'\[\s*recommend(?:ation)?\s*\]', text, flags=re.IGNORECASE)
        text = parts[0]
        text = text.replace('_x000D_', ' ')
        text = re.sub(r'\s+', ' ', text).strip()
        if text and not text.endswith(self.eos_token):
            text += ' ' + self.eos_token

        cleaned = re.sub(r'\[\s*(FINDING|DIAGNOSIS|CONCLUSION)\s*\]', '', text, flags=re.IGNORECASE).strip()
        cleaned = cleaned.replace(self.eos_token, '').strip()

        sentences = [s.strip() for s in re.split(r'\.\s*', cleaned) if s.strip()]
        N = len(sentences)
        if N % 2 == 0 and N > 0:
            half = N // 2
            if all(sentences[i].lower() == sentences[i + half].lower() for i in range(half)):
                final_text = '. '.join(sentences[:half]) + '.'
            else:
                final_text = '. '.join(sentences) + '.'
        else:
            final_text = '. '.join(sentences) + '.'

        final_text = final_text.strip() + ' ' + self.eos_token
        return final_text
    

class FinalSamplesDataset(Dataset):
    def __init__(self, cfg, augment=False, augment_ratio=1):
        """
        Final Samples JSON 기반 Lazy Loading Dataset
        """
        logger = logging.getLogger()
        self.cfg = cfg
        self.img_size = cfg.DATASET.IMAGE_SIZE if hasattr(cfg.DATASET, 'IMAGE_SIZE') else 224
        self.mean = cfg.DATASET.MEAN
        self.std = cfg.DATASET.STD
        self.augment = augment
        self.augment_ratio = augment_ratio
        # albumentations transform 정의
        self.augment_transform = A.Compose([
            A.Resize(self.img_size, self.img_size),
            A.RandomBrightnessContrast(brightness_limit=0.05, contrast_limit=0.05, p=0.3),  # 밝기/대비 약하게
            A.GaussianBlur(blur_limit=(3, 3), p=0.1),  # 블러 약하게, 확률도 낮춤
            A.GaussNoise(var_limit=(1.0, 5.0), p=0.1),  # 노이즈 약하게, 확률도 낮춤
            A.ShiftScaleRotate(shift_limit=0.01, scale_limit=0.02, rotate_limit=5, p=0.15),  # 이동/스케일/회전 약하게
            A.RandomGamma(gamma_limit=(95, 105), p=0.1),  # 감마 변화도 약하게
            A.Normalize(mean=self.mean, std=self.std),
            ToTensorV2(),
        ])
        self.base_transform = A.Compose([
            A.Resize(self.img_size, self.img_size),
            A.Normalize(mean=self.mean, std=self.std),
            ToTensorV2(),
        ])
        # patch_transform은 torchvision transform으로 유지
        from torchvision import transforms
        self.patch_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((112, 112)),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
            transforms.Normalize(mean=self.mean, std=self.std),
        ])

        self.use_raw = cfg.DATASET.USE_RAW
        self.use_patches = cfg.DATASET.USE_PATCH
        self.target_classes = cfg.DATASET.TARGET_CLASSES
        self.use_report = cfg.DATASET.REPORT
        self.concat_patches = cfg.DATASET.CONCAT_PATCH

        if isinstance(self.target_classes, str):
            self.target_classes = self.target_classes.split(",")

        if not isinstance(self.target_classes, list):
            raise TypeError(f"Expected list for target_classes, but got {type(self.target_classes)}")

        self.is_binary = len(self.target_classes) == 2
        self.abnormal_classify = self.is_binary and 'abnormal' in self.target_classes

        self.class_label_mapping = { 0: self.target_classes[0], 1: self.target_classes[1]}

        if self.abnormal_classify:
            self.abnormal_mapping = {'ra': 'abnormal', 'oa': 'abnormal', 'gout': 'abnormal', 'normal': 'normal'}
        else:
            self.abnormal_mapping = None

        # -------------------------------
        # JSON 데이터 로드
        # -------------------------------
        with open(cfg.DATASET.JSON, 'r') as f:
            json_data = json.load(f)

        # JSON -> 내부 데이터 포맷 변환
        self.data = {}
        for idx, item in enumerate(json_data):
            class_label = item.get('class', 'unknown')
            if self.abnormal_mapping:
                class_label = self.abnormal_mapping.get(class_label.lower(), class_label.lower())

            file_path_key = "file_path" if "file_path" in item.keys() else "file_paths"
            
            self.data[idx] = {
                "patient_id": item["patient_id"],
                "file_path": item["merged_image_path"],
                "left_right_file_path": item["file_paths"],
                "class_label": class_label,
                "diagnosis": item.get("diagnosis", ""),
                "keypoints": item.get("keypoints", {}),
                "image_ids": item.get("image_ids", [])
            }
        
        print(f"초기 Initialize self.data length => {len(self.data)}")        
        
        # prepare_data 적용
        if self.is_binary:
            # ✅ 시드 기반 반복 실험일 경우
            print("Prepare Dataset for Binary Classification")
            if hasattr(cfg.DATASET, "SEED") and cfg.DATASET.SEED is not None:
                logger.info(f"[Seed={cfg.DATASET.SEED}] 시드 기반 반복 실험용 샘플링 실행")
                balanced_data, _, _ = prepare_data_with_seed(self.data, self.target_classes, cfg, seed=cfg.DATASET.SEED)
                self.data = {idx: entry for idx, entry in enumerate(balanced_data)}
            else:
                # 기본 처리
                logger.info(f"Basic Prepare Dataset for BC")
                balanced_data, _, _ = prepare_data(self.data, self.target_classes, cfg, self.is_binary)
                self.data = {idx: entry for idx, entry in enumerate(balanced_data)}

        else:
            self.data = self.data
        
        self.indices = list(self.data.keys())
        
        if self.augment:
            self.indices = self.indices * (1 + self.augment_ratio)            

        # bbox json 미리 로드
        bbox_json_path = cfg.DATASET.BBOX_JSON
        with open(bbox_json_path) as f:
            bbox_data = json.load(f)
        bbox_map = {}
        file_path_map = {}
        for data in bbox_data:
            bbox_map.setdefault(data['image_id'], []).append({
                'bbox': data['bbox'],
                'detected_text': data['detected_text']
            })
            file_path_map.setdefault(data['image_id'], []).append(data['file_path'])
        self._bbox_data = bbox_data
        self._bbox_map = bbox_map
        self._file_path_map = file_path_map

    def __len__(self):
        return len(self.indices)

    def get_class_name_from_label(self, label):
        return self.class_label_mapping[label]

    def get_labels(self):
        """데이터셋의 라벨만 빠르게 추출 (이미지 로딩 없이)"""
        labels = []
        for entry in self.data.values():
            label_str = self.abnormal_mapping[entry['class_label'].lower()] if self.abnormal_mapping else entry['class_label'].lower()
            label = self.target_classes.index(label_str)
            labels.append(label)
        return labels


    def __getitem__(self, idx):
        merged_image, meta = self.get_merged_feet_image(idx)
        # merged_image: numpy array (BGR)
        is_aug = self.augment and (self.indices[idx] // len(self.data) > 0)
        if is_aug:
            image = self.augment_transform(image=merged_image)['image']
        else:
            image = self.base_transform(image=merged_image)['image']
        entry = self.data[self.indices[idx] % len(self.data)]
        # patches는 기존대로 생성
        patches = self.generate_patches_from_file_paths(
            entry['left_right_file_path'],
            entry['keypoints']
        )
        patch_tensors = []
        from PIL import Image
        for p in patches:
            patch_tensor = self.patch_transform(Image.fromarray(p).convert("L"))
            patch_tensors.append(patch_tensor)
        if len(patch_tensors) < 34:
            if len(patch_tensors) > 0:
                padding = [torch.zeros_like(patch_tensors[0])] * (34 - len(patch_tensors))
            else:
                padding = [torch.zeros(3, 112, 112)] * 34
            patch_tensors.extend(padding)
        if patch_tensors:
            patch_tensor = torch.stack(patch_tensors, dim=0)
        else:
            patch_tensor = torch.zeros(34, 3, 112, 112)
        label_str = self.abnormal_mapping[entry['class_label'].lower()] if self.abnormal_mapping else entry['class_label'].lower()
        label = self.target_classes.index(label_str)
        if self.is_binary:
            label = torch.tensor(label, dtype=torch.float32).unsqueeze(0)
        else:
            label = torch.tensor(label, dtype=torch.long)
        report = self._clean_report(entry.get("diagnosis", ""))
        if self.concat_patches:
            left_patches = patch_tensors[:17]
            right_patches = patch_tensors[17:]
            patch_tensor = fill_patch_grid(left_patches, right_patches, patch_size=28)
        if self.use_report:
            return image, patch_tensor, label, report, meta
        return image, patch_tensor, label, meta

    def generate_patches_from_file_paths(self, file_paths, keypoints_dict, crop_size=(200, 300), patch_size=(224, 224)):
        # 패치 순서 정의: 1mp, 1pp, 1cib, 2mp, 2pp, 2cib, 3mp, 3pp, 3cib, 4mp, 4pp, 5mp, 5pp, UCB, UNB, DNB, DCB
        patch_order = [
            '1mp', '1pp', '1cib', '2mp', '2pp', '2cib', '3mp', '3pp', '3cib', 
            '4mp', '4pp', '5mp', '5pp', 'UCB', 'UNB', 'DNB', 'DCB'
        ]
        
        def extract_ordered_patches(image, keypoints_side):
            """순서가 보장된 패치 추출"""
            patches = [None] * 17  # 17개 패치를 위한 고정 크기 배열
            
            if not keypoints_side:
                return patches
                
            keypoints = keypoints_side[0]['keypoints']
            
            # 각 keypoint 인덱스에 해당하는 패치 추출
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
                            patches[i] = resized  # 순서대로 저장
                        except:
                            continue
            
            return patches

        def pad_patches(patches):
            """패치 패딩 - 순서 유지"""
            black = np.zeros((patch_size[1], patch_size[0], 3), dtype=np.uint8)
            padded_patches = []
            
            for i in range(17):
                if patches[i] is not None:
                    padded_patches.append(patches[i])
                else:
                    padded_patches.append(black)
            
            return padded_patches

        left_patches, right_patches = [], []

        if len(file_paths) == 1:
            # merged image 하나
            image = cv2.imread(file_paths[0])
            if image is None:
                raise FileNotFoundError(f"Cannot read image at {file_paths[0]}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            if 'left' in keypoints_dict and keypoints_dict['left']:
                left_patches = extract_ordered_patches(image, keypoints_dict['left'])

            if 'right' in keypoints_dict and keypoints_dict['right']:
                right_patches = extract_ordered_patches(image, keypoints_dict['right'])

        elif len(file_paths) == 2:
            # left image
            left_image = cv2.imread(file_paths[0])
            if left_image is None:
                raise FileNotFoundError(f"Cannot read left image at {file_paths[0]}")
            left_image = cv2.cvtColor(left_image, cv2.COLOR_BGR2RGB)

            # right image
            right_image = cv2.imread(file_paths[1])
            if right_image is None:
                raise FileNotFoundError(f"Cannot read right image at {file_paths[1]}")
            right_image = cv2.cvtColor(right_image, cv2.COLOR_BGR2RGB)

            if 'left' in keypoints_dict and keypoints_dict['left']:
                left_patches = extract_ordered_patches(left_image, keypoints_dict['left'])

            if 'right' in keypoints_dict and keypoints_dict['right']:
                right_patches = extract_ordered_patches(right_image, keypoints_dict['right'])

        # fallback 처리 - 순서 유지하면서 처리
        if left_patches and not right_patches:
            right_patches = [cv2.flip(p, 1) if p is not None else None for p in left_patches]
        elif right_patches and not left_patches:
            left_patches = [cv2.flip(p, 1) if p is not None else None for p in right_patches]
        elif not left_patches and not right_patches:
            black = np.zeros((patch_size[1], patch_size[0], 3), dtype=np.uint8)
            return [black] * 34

        # 패딩 처리 - 순서 유지
        left_patches = pad_patches(left_patches)
        right_patches = pad_patches(right_patches)

        # 최종 순서: left_patches (17개) + right_patches (17개) = 34개
        return left_patches + right_patches



    def _clean_report(self, text):
        import re, unicodedata
        eos_token = "<eos>"
        text = unicodedata.normalize('NFKC', text)
        text = re.sub(r'[^\x00-\x7F]+', ' ', text)
        text = re.sub(r'([.!?]){2,}', r'\1', text)
        text = re.sub(r'\[\s*finding\s*\]', '[FINDING]', text, flags=re.IGNORECASE)
        text = re.sub(r'\[\s*conclusion\s*\]', '[CONCLUSION]', text, flags=re.IGNORECASE)
        text = re.sub(r'\[\s*diagnosis\s*\]', '[DIAGNOSIS]', text, flags=re.IGNORECASE)
        parts = re.split(r'\[\s*recommend(?:ation)?\s*\]', text, flags=re.IGNORECASE)
        text = parts[0]
        text = text.replace('_x000D_', ' ')
        text = re.sub(r'\s+', ' ', text).strip()
        if text and not text.endswith(eos_token):
            text += ' ' + eos_token

        cleaned = re.sub(r'\[\s*(FINDING|DIAGNOSIS|CONCLUSION)\s*\]', '', text, flags=re.IGNORECASE).strip()
        cleaned = cleaned.replace(eos_token, '').strip()

        sentences = [s.strip() for s in re.split(r'\.\s*', cleaned) if s.strip()]
        if len(sentences) % 2 == 0 and all(sentences[i].lower() == sentences[i + len(sentences)//2].lower() for i in range(len(sentences)//2)):
            sentences = sentences[:len(sentences)//2]
        final_text = '. '.join(sentences) + '.'
        final_text = final_text.strip() + ' ' + eos_token
        return final_text

    def analyze_original_image_sizes(self, max_samples=None):
        """
        transform 적용 전 전체 이미지 해상도(min, max, mean) 출력
        Args:
            max_samples: 최대 샘플 수 (None이면 전체)
        Returns:
            sizes: (width, height) 튜플 리스트
        """
        from PIL import Image
        import numpy as np
        sizes = []
        data_list = list(self.data.values())
        if max_samples is not None:
            data_list = data_list[:max_samples]
        for entry in data_list:
            file_path = entry['file_path']
            try:
                with Image.open(file_path) as img:
                    sizes.append(img.size)  # (width, height)
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
        if sizes:
            widths, heights = zip(*sizes)
            widths = np.array(widths)
            heights = np.array(heights)
            print(f"총 {len(sizes)}개 샘플 분석")
            print(f"최소 해상도: {widths.min()}x{heights.min()}")
            print(f"최대 해상도: {widths.max()}x{heights.max()}")
            print(f"평균 해상도: {widths.mean():.1f}x{heights.mean():.1f}")
        else:
            print("이미지 정보를 읽을 수 없습니다.")
        return sizes

    def analyze_channels(self, max_samples=50):
        """
        데이터셋의 이미지들을 분석하여 채널 정보를 확인
        
        Args:
            max_samples: 분석할 최대 샘플 수
            
        Returns:
            dict: 분석 결과
        """
        print(f"FinalSamplesDataset 채널 분석 시작 (최대 {max_samples}개 샘플)")
        
        results = []
        grayscale_count = 0
        non_grayscale_count = 0
        
        # 데이터를 리스트로 변환하여 인덱싱
        data_list = list(self.data.values())
        
        for i, entry in enumerate(tqdm(data_list[:max_samples], total=min(max_samples, len(data_list)))):
            file_path = entry['file_path']
            result = check_image_channels(file_path)
            results.append(result)
            
            if 'error' not in result:
                if result['is_grayscale']:
                    grayscale_count += 1
                else:
                    non_grayscale_count += 1
        
        # 결과 분석
        print(f"\n=== 채널 분석 결과 ===")
        print(f"총 분석된 이미지: {len(results)}")
        print(f"Grayscale 이미지: {grayscale_count}")
        print(f"Non-grayscale 이미지: {non_grayscale_count}")
        print(f"Grayscale 비율: {grayscale_count/len(results)*100:.2f}%")
        
        # Non-grayscale 이미지들의 상세 정보
        non_grayscale_results = [r for r in results if 'error' not in r and not r['is_grayscale']]
        
        if non_grayscale_results:
            print(f"\n=== Non-grayscale 이미지 상세 정보 (처음 5개) ===")
            for i, result in enumerate(non_grayscale_results[:5]):
                print(f"\n{i+1}. {os.path.basename(result['file_path'])}")
                print(f"   원본 모드: {result['original_mode']}")
                print(f"   R-G 최대 차이: {result['r_g_max_diff']}")
                print(f"   R-B 최대 차이: {result['r_b_max_diff']}")
                print(f"   G-B 최대 차이: {result['g_b_max_diff']}")
                print(f"   R 평균: {result['r_mean']:.2f}")
                print(f"   G 평균: {result['g_mean']:.2f}")
                print(f"   B 평균: {result['b_mean']:.2f}")
        
        # 에러가 있는 이미지들
        error_results = [r for r in results if 'error' in r]
        if error_results:
            print(f"\n=== 에러가 발생한 이미지들 ===")
            for result in error_results:
                print(f"  {os.path.basename(result['file_path'])}: {result['error']}")
        
        return {
            'total_analyzed': len(results),
            'grayscale_count': grayscale_count,
            'non_grayscale_count': non_grayscale_count,
            'grayscale_ratio': grayscale_count/len(results)*100 if len(results) > 0 else 0,
            'results': results
        }

    # def get_merged_feet_image(self, idx):
    #     """
    #     환자별로 좌/우 발 bbox만 crop해서 결합된 이미지를 반환
    #     Args:
    #         idx: self.indices 기준 인덱스
    #     Returns:
    #         merged_image (np.ndarray), meta (dict)
    #     """
    #     bbox_map = self._bbox_map
    #     file_path_map = self._file_path_map
    #     data_idx = self.indices[idx] % len(self.data)
    #     entry = self.data[data_idx]
    #     image_ids = entry['image_ids']

    #     left_img_id = entry['keypoints']['left'][0]['image_id']
    #     right_img_id = entry['keypoints']['right'][0]['image_id']

    #     # left foot image_id, right foot image_id 
    #     # print(f"left foot image_id: {left_img_id}, right foot image_id: {right_img_id}")

    #     meta = {
    #         'patient_id': entry['patient_id'],
    #         'class_label': entry['class_label']
    #     }
    #     crops = []
    #     def resize_keep_aspect(img, target_h):
    #         h, w = img.shape[:2]
    #         scale = target_h / h
    #         new_w = int(w * scale)
    #         return cv2.resize(img, (new_w, target_h), interpolation=cv2.INTER_AREA)
    #     # 2장: keypoints의 left/right의 image_id로 좌/우 발 구분
    #     def euclidean(p1, p2):
    #         return np.linalg.norm(np.array(p1) - np.array(p2))

    #     all_combined = 0
    #     merged_from_one_image = 0
    #     merged_from_two_image = 0

    #     if left_img_id == right_img_id:
    #         # print("two boxes in one image == cropping")
    #         file_path = file_path_map[left_img_id][0]
    #         image = cv2.imread(file_path)
    #         _img_id = left_img_id
    #         bboxes = bbox_map[_img_id]

    #         # keypoint 중심
    #         left_kpt_center = entry['keypoints']['left'][0]['center']
    #         right_kpt_center = entry['keypoints']['right'][0]['center']

    #         bbox_centers = []
    #         for bbox in bboxes:
    #             x, y, w, h = bbox
    #             bbox_center = [x + w // 2, y + h // 2]
    #             bbox_centers.append(bbox_center)

    #         # 거리 계산
    #         left_distances = [euclidean(bc, left_kpt_center) for bc in bbox_centers]
    #         right_distances = [euclidean(bc, right_kpt_center) for bc in bbox_centers]

    #         # 최소 거리 기준으로 bbox 순서 매칭
    #         left_bbox_idx = int(np.argmin(left_distances))
    #         right_bbox_idx = 1 - left_bbox_idx

    #         # bbox를 좌/우 순으로 정렬
    #         left_bbox = bboxes[left_bbox_idx]
    #         right_bbox = bboxes[right_bbox_idx]

    #         # print(f"[{_img_id}] Left bbox: {left_bbox}, Right bbox: {right_bbox}")
    #         x, y, w, h = map(int, left_bbox)
    #         left_crop = resize_keep_aspect(image[y:y+h, x:x+w], target_h=1600)

    #         x, y, w, h = map(int, right_bbox)
    #         right_crop = resize_keep_aspect(image[y:y+h, x:x+w], target_h=1600)

    #         merged = np.concatenate([left_crop, right_crop], axis=1)
    #         merged_from_one_image += 1
                        
    #     else:
    #         # print("have box for each image == cropping ")
    #         # 다르다는 것은 각각 서로 다른 이미지 상에 bbox가 존재
    #         left_file_path = file_path_map[left_img_id][0]
    #         right_file_path = file_path_map[right_img_id][0]
    #         left_image = cv2.imread(left_file_path)
    #         right_image = cv2.imread(right_file_path)
    #         left_bbox = bbox_map[left_img_id][0]
    #         right_bbox = bbox_map[right_img_id][0]
    #         # print(f"left_bbox {left_bbox} || right_bbox {right_bbox}")


    #         left_crop = self._crop_image(left_image, left_bbox)
    #         right_crop = self._crop_image(right_image, right_bbox)
    #         left_crop = resize_keep_aspect(left_crop, target_h=1600)
    #         right_crop = resize_keep_aspect(right_crop, target_h=1600)
    #         merged = np.concatenate([left_crop, right_crop], axis=1)
    #         merged_from_two_image += 1
        
    #     all_combined = merged_from_one_image + merged_from_two_image

    #     meta['after_merged'] = all_combined
        

    #     return merged, meta

    def get_merged_feet_image(self, idx):
        """
        환자별로 좌/우 발 bbox만 crop해서 결합된 이미지를 반환
        Args:
            idx: self.indices 기준 인덱스
        Returns:
            merged_image (np.ndarray), meta (dict)
        """
        bbox_map = self._bbox_map
        file_path_map = self._file_path_map
        data_idx = self.indices[idx] % len(self.data)
        entry = self.data[data_idx]
        image_ids = entry['image_ids']

        left_img_id = entry['keypoints']['left'][0]['image_id']
        right_img_id = entry['keypoints']['right'][0]['image_id']

        # left foot image_id, right foot image_id 
        # print(f"left foot image_id: {left_img_id}, right foot image_id: {right_img_id}")

        meta = {
            'patient_id': entry['patient_id'],
            'class_label': entry['class_label']
        }
        crops = []
        
        # merge한 이미지 전체에 대해 배경 제거하는 함수
        def remove_background_from_merged(merged_image, threshold=30):
            """
            merge된 발 이미지 전체에서 배경을 제거하는 함수
            Args:
                merged_image: merge된 발 이미지 (BGR)
                threshold: 배경 제거 임계값 (0-255)
            Returns:
                masked_image: 배경이 제거된 이미지
            """
            # BGR to HSV 변환
            hsv = cv2.cvtColor(merged_image, cv2.COLOR_BGR2HSV)
            
            # 밝기 채널(V)을 사용하여 어두운 배경 제거
            _, mask = cv2.threshold(hsv[:, :, 2], threshold, 255, cv2.THRESH_BINARY)
            
            # 마스크를 3채널로 확장
            mask_3d = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            
            # 마스크 적용
            masked_image = cv2.bitwise_and(merged_image, mask_3d)
            
            return masked_image
        
        # 마진과 패딩을 추가하는 함수 (원본 크기 유지, 가운데 마진 없음)
        def add_padding_only(left_crop, right_crop, padding=30):
            """
            좌/우 발 이미지에 패딩만 추가 (가운데 마진 없음)
            원본 crop 크기는 유지
            Args:
                left_crop: 왼쪽 발 crop 이미지
                right_crop: 오른쪽 발 crop 이미지
                padding: 바깥 가장자리 패딩 (픽셀)
            Returns:
                merged_image: 패딩이 추가된 결합 이미지
            """
            left_h, left_w = left_crop.shape[:2]
            right_h, right_w = right_crop.shape[:2]
            
            # 패딩만 포함된 최종 이미지 크기 계산 (가운데 마진 없음)
            final_width = left_w + right_w + (padding * 2)
            final_height = max(left_h, right_h) + (padding * 2)
            
            # 검은색 배경으로 초기화
            merged = np.zeros((final_height, final_width, 3), dtype=np.uint8)
            
            # 왼쪽 발 배치 (패딩만큼 오프셋)
            y_offset = padding
            x_offset = padding
            merged[y_offset:y_offset+left_h, x_offset:x_offset+left_w] = left_crop
            
            # 오른쪽 발 배치 (왼쪽 발 바로 옆에 배치, 마진 없음)
            x_offset = padding + left_w
            merged[y_offset:y_offset+right_h, x_offset:x_offset+right_w] = right_crop
            
            return merged
        
        # 2장: keypoints의 left/right의 image_id로 좌/우 발 구분
        def euclidean(p1, p2):
            return np.linalg.norm(np.array(p1) - np.array(p2))

        all_combined = 0
        merged_from_one_image = 0
        merged_from_two_image = 0

        if left_img_id == right_img_id:
            # print("two boxes in one image == cropping")
            file_path = file_path_map[left_img_id][0]
            image = cv2.imread(file_path)
            _img_id = left_img_id
            bbox_infos = bbox_map[_img_id]
            
            # detected_text 기준으로 bbox 분류
            left_bbox = None
            right_bbox = None
            
            for bbox_info in bbox_infos:
                if bbox_info['detected_text'] == 'L':
                    left_bbox = bbox_info['bbox']
                elif bbox_info['detected_text'] == 'R':
                    right_bbox = bbox_info['bbox']
            
            # bbox가 제대로 분류되지 않은 경우 fallback
            if left_bbox is None or right_bbox is None:
                # keypoint 중심과의 거리로 fallback
                left_kpt_center = entry['keypoints']['left'][0]['center']
                right_kpt_center = entry['keypoints']['right'][0]['center']
                
                bbox_centers = []
                for bbox_info in bbox_infos:
                    x, y, w, h = bbox_info['bbox']
                    bbox_center = [x + w // 2, y + h // 2]
                    bbox_centers.append(bbox_center)
                
                # side 값 결정 (첫 번째 bbox의 detected_text 사용)
                side = bbox_infos[0]['detected_text']
                
                # side 값에 따라 좌/우 발(손) 매칭 방식 결정
                # side가 "R"이면, 이미지에서 본 left가 실제로는 오른쪽, 나머지가 왼쪽
                if side == "R":
                    # left_kpt_center에 더 가까운 bbox가 실제로는 오른쪽(오른손/오른발)
                    left_distances = [euclidean(bc, left_kpt_center) for bc in bbox_centers]
                    right_distances = [euclidean(bc, right_kpt_center) for bc in bbox_centers]
                    right_bbox_idx = int(np.argmin(left_distances))
                    left_bbox_idx = 1 - right_bbox_idx
                else:
                    # left_kpt_center에 더 가까운 bbox가 실제로는 왼쪽(왼손/왼발)
                    left_distances = [euclidean(bc, left_kpt_center) for bc in bbox_centers]
                    right_distances = [euclidean(bc, right_kpt_center) for bc in bbox_centers]
                    left_bbox_idx = int(np.argmin(left_distances))
                    right_bbox_idx = 1 - left_bbox_idx
                
                left_bbox = bbox_infos[left_bbox_idx]['bbox']
                right_bbox = bbox_infos[right_bbox_idx]['bbox']

            # print(f"[{_img_id}] Left bbox: {left_bbox}, Right bbox: {right_bbox}")
            x, y, w, h = map(int, left_bbox)
            left_crop = image[y:y+h, x:x+w]  # 원본 크기 유지

            x, y, w, h = map(int, right_bbox)
            right_crop = image[y:y+h, x:x+w]  # 원본 크기 유지

            # 먼저 merge
            merged = add_padding_only(left_crop, right_crop, padding=30)
            
            # merge한 뒤 전체 이미지에 대해 배경 제거
            merged = remove_background_from_merged(merged, threshold=30)
            
            merged_from_one_image += 1
                        
        else:
            # print("have box for each image == cropping ")
            # 다르다는 것은 각각 서로 다른 이미지 상에 bbox가 존재
            left_file_path = file_path_map[left_img_id][0]
            right_file_path = file_path_map[right_img_id][0]
            left_image = cv2.imread(left_file_path)
            right_image = cv2.imread(right_file_path)
            
            # 새로운 bbox_map 구조에 맞게 bbox 추출
            left_bbox = bbox_map[left_img_id][0]['bbox']
            right_bbox = bbox_map[right_img_id][0]['bbox']
            # print(f"left_bbox {left_bbox} || right_bbox {right_bbox}")

            left_crop = self._crop_image(left_image, left_bbox)
            right_crop = self._crop_image(right_image, right_bbox)
            
            # 먼저 merge
            merged = add_padding_only(left_crop, right_crop, padding=30)
            
            # merge한 뒤 전체 이미지에 대해 배경 제거
            merged = remove_background_from_merged(merged, threshold=30)
            
            merged_from_two_image += 1
        
        all_combined = merged_from_one_image + merged_from_two_image

        meta['after_merged'] = all_combined
        

        return merged, meta, left_img_id == right_img_id

    def _crop_image(self, img, bbox):
        x, y, w, h = map(int, bbox)
        return img[y:y+h, x:x+w]


def fill_patch_grid(left_patches, right_patches, patch_size=28):

    # device/dtype 안전하게 가져오기
    if len(left_patches) > 0 and isinstance(left_patches[0], torch.Tensor):
        device = left_patches[0].device
        dtype = left_patches[0].dtype
    else:
        device = torch.device('cpu')
        dtype = torch.float32
    black_patch = torch.zeros(3, patch_size, patch_size, device=device, dtype=dtype)
    # 인덱스 매핑
    left_dict = {
        'L_4mp': 9, 'L_3mp': 6, 'L_2mp': 3, 'L_1mp': 0,
        'L_4pp': 10, 'L_3pp': 7, 'L_2pp': 4, 'L_1pp': 1,
        'L_5mp': 11, 'L_3CiB': 8, 'L_2CiB': 5, 'L_1CiB': 2,
        'L_5pp': 12, 'L_UCB': 13, 'L_UNB': 14,
        'L_DCB': 15, 'L_DNB': 16
    }
    right_dict = {
        'R_1mp': 0, 'R_2mp': 3, 'R_3mp': 6, 'R_4mp': 9,
        'R_1pp': 1, 'R_2pp': 4, 'R_3pp': 7, 'R_4pp': 10,
        'R_1CiB': 2, 'R_2CiB': 5, 'R_3CiB': 8, 'R_5mp': 11,
        'R_UNB': 13, 'R_UCB': 14, 'R_5pp': 12,
        'R_DNB': 15
    }
    # 1채널 패치를 3채널로 변환
    def to_three_channel(patch):
        if patch.shape[0] == 1:
            return patch.repeat(3, 1, 1)
        return patch
    # 패치 크기를 patch_size x patch_size로 맞춤
    def resize_patch(patch):
        if patch.shape[1] != patch_size or patch.shape[2] != patch_size:
            patch = F.interpolate(patch.unsqueeze(0), size=(patch_size, patch_size), mode='bilinear', align_corners=False).squeeze(0)
        return patch
    left_patches = [resize_patch(to_three_channel(p)) for p in left_patches]
    right_patches = [resize_patch(to_three_channel(p)) for p in right_patches]
    # 레이아웃 정의 (5x8, 아래 3줄은 제거)
    layout = [
        ['L_4mp', 'L_3mp', 'L_2mp', 'L_1mp', 'R_1mp', 'R_2mp', 'R_3mp', 'R_4mp'],
        ['L_4pp', 'L_3pp', 'L_2pp', 'L_1pp', 'R_1pp', 'R_2pp', 'R_3pp', 'R_4pp'],
        ['L_5mp', 'L_3CiB', 'L_2CiB', 'L_1CiB', 'R_1CiB', 'R_2CiB', 'R_3CiB', 'R_5mp'],
        ['L_5pp', 'L_UCB', 'L_UNB', 0, 0, 'R_UNB', 'R_UCB', 'R_5pp'],
        [0, 'L_DCB', 'L_DNB', 0, 0, 'R_DNB', 'R_DNB', 0],
    ]  # 5x8 grid
    grid_rows = []
    for row in layout:
        row_patches = []
        for name in row:
            if name == 0 or name is None:
                row_patches.append(black_patch)
            elif isinstance(name, str) and name.startswith('L_'):
                row_patches.append(left_patches[left_dict[name]])
            elif isinstance(name, str) and name.startswith('R_'):
                row_patches.append(right_patches[right_dict[name]])
            else:
                row_patches.append(black_patch)
        row_cat = torch.cat(row_patches, dim=2)  # width 방향
        grid_rows.append(row_cat)
    grid_img = torch.cat(grid_rows, dim=1)  # height 방향
    # 마지막에 224x224로 리사이즈
    grid_img = F.interpolate(grid_img.unsqueeze(0), size=(224, 224), mode='bilinear', align_corners=False).squeeze(0)
    return grid_img  # shape: (3, 224, 224)

def fill_and_pad_patch_grid(left_patches, right_patches, patch_size=28, final_size=224):
    grid = fill_patch_grid(left_patches, right_patches, patch_size=patch_size)
    pad_height = final_size - grid.shape[1]
    if pad_height > 0:
        grid = F.pad(grid, (0, 0, 0, pad_height))
    return grid


