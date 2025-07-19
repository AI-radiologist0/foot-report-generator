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


from utils.utils import prepare_binary_data, prepare_data, prepare_data_with_seed

# 3채널용 transform (원본 이미지)
image_transform_rgb = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# 1채널용 transform (패치)
def get_patch_transform_gray(concat_patch):
    size = (28, 28) if concat_patch else (112, 112)
    return transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
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
    def __init__(self, cfg, image_transform=image_transform_rgb, patch_transform=None, augment_transform=None, augment_ratio=1):
        """
        Final Samples JSON 기반 Lazy Loading Dataset
        """
        logger = logging.getLogger()
        self.cfg = cfg
        self.image_transform = image_transform  # 3채널용
        self.patch_transform = patch_transform if patch_transform is not None else get_patch_transform_gray(cfg.DATASET.CONCAT_PATCH) # 1채널용
        self.augment_transform = augment_transform
        self.augment_ratio = augment_ratio

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
                "left_right_file_path": item["file_paths"],  # 추가!
                "class_label": class_label,
                "diagnosis": item.get("diagnosis", ""),
                "keypoints": item.get("keypoints", {})  # left/right 모두
            }
        
        
        
        # prepare_data 적용
        if self.is_binary:
            # ✅ 시드 기반 반복 실험일 경우
            if hasattr(cfg.DATASET, "SEED") and cfg.DATASET.SEED is not None:
                logger.info(f"[Seed={cfg.DATASET.SEED}] 시드 기반 반복 실험용 샘플링 실행")
                balanced_data, _, _ = prepare_data_with_seed(self.data, self.target_classes, cfg, seed=cfg.DATASET.SEED)
                self.data = {idx: entry for idx, entry in enumerate(balanced_data)}
            else:
                # 기본 처리
                balanced_data, _, _ = prepare_data(self.data, self.target_classes, cfg, self.is_binary)
                self.data = {idx: entry for idx, entry in enumerate(balanced_data)}

        else:
            self.data = self.data
        
        self.total_len = len(self.data) * (1 + self.augment_ratio)


    def __len__(self):
        return self.total_len

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


    def __getitem__(self, idx):
        # 원본 or 증강본 선택
        data_idx = idx % len(self.data)
        is_aug = idx // len(self.data) > 0

        entry = self.data[data_idx]
        image = Image.open(entry['file_path']).convert("RGB")
        if is_aug and self.augment_transform is not None:
            image = self.augment_transform(image)
        else:
            image = self.image_transform(image)

        # patches는 따로 생성
        patches = self.generate_patches_from_file_paths(
            entry['left_right_file_path'],
            entry['keypoints']
        )

        # patch transform
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

        # 레이블 변환
        label_str = self.abnormal_mapping[entry['class_label'].lower()] if self.abnormal_mapping else entry['class_label'].lower()
        label = self.target_classes.index(label_str)

        if self.is_binary:
            label = torch.tensor(label, dtype=torch.float32).unsqueeze(0)
        else:
            label = torch.tensor(label, dtype=torch.long)

        # 리포트
        report = self._clean_report(entry.get("diagnosis", ""))
        
        # 메타 정보 구성
        meta = {
            "patient_id": entry['patient_id'],
            "class_label": entry['class_label']
        }
        
        if self.concat_patches:
            # patch_tensors: (34, 1, patch_size, patch_size) 또는 list
            left_patches = patch_tensors[:17]
            right_patches = patch_tensors[17:]
            if isinstance(left_patches, list):
                left_patches = torch.stack(left_patches, dim=0)
            if isinstance(right_patches, list):
                right_patches = torch.stack(right_patches, dim=0)
            patch_tensor = fill_and_pad_patch_grid(left_patches, right_patches, patch_size=28, final_size=224)
            # 1채널 → 3채널 복제
            if patch_tensor.shape[0] == 1:
                patch_tensor = patch_tensor.repeat(3, 1, 1)
        else:
            patch_tessor = patch_tensor

        if self.use_report:
            return image, patch_tensor, label, report, meta
        return image, patch_tensor, label, meta

    def generate_patches_from_file_paths(self, file_paths, keypoints_dict, crop_size=(200, 300), patch_size=(224, 224)):
        def extract(image, keypoints_side):
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

        def pad(patches):
            black = np.zeros((patch_size[1], patch_size[0], 3), dtype=np.uint8)
            while len(patches) < 17:
                patches.append(black)
            return patches[:17]

        left_patches, right_patches = [], []

        if len(file_paths) == 1:
            # merged image 하나
            image = cv2.imread(file_paths[0])
            if image is None:
                raise FileNotFoundError(f"Cannot read image at {file_paths[0]}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            if 'left' in keypoints_dict and keypoints_dict['left']:
                left_patches = extract(image, keypoints_dict['left'])

            if 'right' in keypoints_dict and keypoints_dict['right']:
                right_patches = extract(image, keypoints_dict['right'])

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
                left_patches = extract(left_image, keypoints_dict['left'])

            if 'right' in keypoints_dict and keypoints_dict['right']:
                right_patches = extract(right_image, keypoints_dict['right'])

        # fallback 처리
        if left_patches and not right_patches:
            right_patches = [cv2.flip(p, 1) for p in left_patches]
        elif right_patches and not left_patches:
            left_patches = [cv2.flip(p, 1) for p in right_patches]
        elif not left_patches and not right_patches:
            black = np.zeros((patch_size[1], patch_size[0], 3), dtype=np.uint8)
            return [black] * 34

        left_patches = pad(left_patches)
        right_patches = pad(right_patches)

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


def fill_patch_grid(left_patches, right_patches, patch_size=28):
    grid = torch.zeros((1, 8*patch_size, 5*patch_size), dtype=left_patches.dtype, device=left_patches.device)

    # left
    left_indices = [
        (0,0), (1,0), (2,0), (0,1), (1,1), (2,1), (0,2), (1,2), (2,2), (0,3), (1,3), (0,4), (1,4)
    ]
    # UCB, UNB는 같은 패치를 두 칸에 넣음
    # left_patches[13] = UCB, [14] = UNB, [15]=DNB, [16]=DCB
    grid[:, 2*patch_size:3*patch_size, 3*patch_size:4*patch_size] = left_patches[13]  # UCB
    grid[:, 2*patch_size:3*patch_size, 4*patch_size:5*patch_size] = left_patches[13]  # UCB
    grid[:, 3*patch_size:4*patch_size, 0*patch_size:1*patch_size] = left_patches[14]  # UNB
    grid[:, 3*patch_size:4*patch_size, 1*patch_size:2*patch_size] = left_patches[14]  # UNB
    grid[:, 3*patch_size:4*patch_size, 2*patch_size:3*patch_size] = left_patches[15]  # DNB
    grid[:, 3*patch_size:4*patch_size, 3*patch_size:4*patch_size] = left_patches[16]  # DCB

    for i, (r, c) in enumerate(left_indices):
        grid[:, r*patch_size:(r+1)*patch_size, c*patch_size:(c+1)*patch_size] = left_patches[i]

    # right
    right_indices = [
        (4,0), (5,0), (6,0), (4,1), (5,1), (6,1), (4,2), (5,2), (6,2), (4,3), (5,3), (4,4), (5,4)
    ]
    grid[:, 6*patch_size:7*patch_size, 3*patch_size:4*patch_size] = right_patches[13]  # UCB
    grid[:, 6*patch_size:7*patch_size, 4*patch_size:5*patch_size] = right_patches[13]  # UCB
    grid[:, 7*patch_size:8*patch_size, 0*patch_size:1*patch_size] = right_patches[14]  # UNB
    grid[:, 7*patch_size:8*patch_size, 1*patch_size:2*patch_size] = right_patches[14]  # UNB
    grid[:, 7*patch_size:8*patch_size, 2*patch_size:3*patch_size] = right_patches[15]  # DNB
    grid[:, 7*patch_size:8*patch_size, 3*patch_size:4*patch_size] = right_patches[16]  # DCB

    for i, (r, c) in enumerate(right_indices):
        grid[:, r*patch_size:(r+1)*patch_size, c*patch_size:(c+1)*patch_size] = right_patches[i]

    return grid

def fill_and_pad_patch_grid(left_patches, right_patches, patch_size=28, final_size=224):
    grid = fill_patch_grid(left_patches, right_patches, patch_size=patch_size)
    pad_height = final_size - grid.shape[1]
    if pad_height > 0:
        grid = F.pad(grid, (0, 0, 0, pad_height))
    return grid


