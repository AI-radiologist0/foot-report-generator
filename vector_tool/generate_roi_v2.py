from pycocomedical import COCOMedical

import cv2
import numpy as np
import os
import pickle
from tqdm import tqdm

# ------- config -------
PATCH_SIZE = (224, 224)
CROP_SIZE = (200, 300)
MAX_PATCHES = 34
SAVE_PATH = "data/pkl/final_output_left_right_ordered.pkl"
# ----------------------

def extract_patches_from_keypoints(image, keypoints_side, patch_size=PATCH_SIZE, crop_size=CROP_SIZE):
    patches = []
    height, width, _ = image.shape
    keypoints = keypoints_side[0]['keypoints']
    for i in range(17):
        x, y, score = int(keypoints[i * 3]), int(keypoints[i * 3 + 1]), keypoints[i * 3 + 2]
        if score > 0.0:
            x_min = max(x - crop_size[0] // 2, 0)
            y_min = max(y - crop_size[1] // 2, 0)
            x_max = min(x + crop_size[0] // 2, width)
            y_max = min(y + crop_size[1] // 2, height)
            crop = image[y_min:y_max, x_min:x_max]
            if crop.size > 0:
                try:
                    resized = cv2.resize(crop, patch_size)
                    patches.append(resized)
                except:
                    continue
    return patches

def create_black_patch(size=PATCH_SIZE):
    return np.zeros((size[1], size[0], 3), dtype=np.uint8)

def flip_patch_images(patches):
    return [cv2.flip(p, 1) for p in patches]

def pad_or_trim(patch_list, target_len=17):
    while len(patch_list) < target_len:
        patch_list.append(create_black_patch())
    return patch_list[:target_len]

def process_disease_entry(entry):
    image_path = entry['file_path']
    image_id = entry['image_id']
    try:
        image = cv2.imread(image_path)
        if image is None:
            print(f"⚠️ Cannot load image: {image_path}")
            return None
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    except Exception as e:
        print(f"⚠️ Error reading image {image_path}: {e}")
        return None

    keypoints_info = entry.get('keypoint_id', {})
    left_patches, right_patches = [], []

    if 'left' in keypoints_info and keypoints_info['left']:
        left_patches = extract_patches_from_keypoints(image, keypoints_info['left'])

    if 'right' in keypoints_info and keypoints_info['right']:
        right_patches = extract_patches_from_keypoints(image, keypoints_info['right'])

    # 한쪽만 있는 경우 → flip하여 다른 쪽 보충
    if left_patches and not right_patches:
        right_patches = flip_patch_images(left_patches)

    elif right_patches and not left_patches:
        left_patches = flip_patch_images(right_patches)

    elif not left_patches and not right_patches:
        return None  # skip if both missing

    # 패치 수 보정
    left_patches = pad_or_trim(left_patches, 17)
    right_patches = pad_or_trim(right_patches, 17)

    patches = left_patches + right_patches  # 순서 보장

    return {
        "image_id": image_id,
        "patient_id": entry.get("patient_id", ""),
        "file_path": image_path,
        "class": entry.get("class_label", ""),
        "diagnosis": entry.get("diagnosis", ""),
        "bbx": patches  # 총 34개
    }


# --------- 메인 처리 루프 ---------

coco_medical = COCOMedical()
coco_medical.load_json("data/merge/output.json")

# 2. annotations -> Disease 객체 변환
# annotations = coco_medical.dataset.get("annotations", [])
disease_list = []
for idx, value in enumerate(coco_medical.diseases):
    disease_list.append(coco_medical.diseases[value].to_dict())

final_data = {}
for entry in tqdm(disease_list, desc="Processing disease entries"):
    result = process_disease_entry(entry)
    if result is not None:
        final_data[result['image_id']] = result

print(f"✅ 완료: 총 {len(final_data)}개 이미지 처리됨.")

# --------- Pickle 저장 ---------
os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
with open(SAVE_PATH, 'wb') as f:
    pickle.dump(final_data, f)

print(f"📦 저장 완료: {SAVE_PATH}")
