import os
import matplotlib.pyplot as plt
import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

import _init_path
from dataset.cocomedical import COCOMedicalDataset  # 이전에 만든 Dataset 클래스

# 데이터셋 로드
json_path = "data/json/merge/output.json"
dataset = COCOMedicalDataset(json_path, bbox_size=120)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

# 저장할 폴더 경로
output_dir = "vis_test"
os.makedirs(output_dir, exist_ok=True)

def visualize_and_save_samples(dataloader, num_samples=4):
    for i, sample in enumerate(dataloader):
        if i >= num_samples:
            break

        image_id = sample["image_id"]

        # 🔹 keypoints를 리스트 형태로 유지 (내부 요소가 Tensor면 변환)
        left_keypoints = [kp.tolist() if isinstance(kp, torch.Tensor) else kp for kp in sample["left_keypoints"]]
        right_keypoints = [kp.tolist() if isinstance(kp, torch.Tensor) else kp for kp in sample["right_keypoints"]]

        # 🔹 patches를 numpy 배열로 변환
        left_patches = [p.numpy() if isinstance(p, torch.Tensor) else p for p in sample["left_patches"]]
        right_patches = [p.numpy() if isinstance(p, torch.Tensor) else p for p in sample["right_patches"]]

        # 원본 이미지 로드
        image_info = dataset.coco.get_image(image_id)
        image = cv2.imread(image_info.file_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        # 1️⃣ 원본 이미지 표시
        axes[1].imshow(image)
        axes[1].set_title(f"Original Image (ID: {image_id})")
        axes[1].axis("off")

        # 🔴 Left keypoints 표시 (빨간색)
        for j in range(0, len(left_keypoints), 3):
            x, y, score = int(left_keypoints[j]), int(left_keypoints[j+1]), left_keypoints[j+2]
            if score > 0.0:
                axes[1].scatter(x, y, color='red', s=50)

        # 🔵 Right keypoints 표시 (파란색)
        for j in range(0, len(right_keypoints), 3):
            x, y, score = int(right_keypoints[j]), int(right_keypoints[j+1]), right_keypoints[j+2]
            if score > 0.0:
                axes[1].scatter(x, y, color='blue', s=50)

        # 2️⃣ Left patches 표시
        combined_left_patches = np.hstack([left_patches[j] for j in range(17)])
        axes[0].imshow(combined_left_patches)
        axes[0].set_title("Left Patches")
        axes[0].axis("off")

        # 3️⃣ Right patches 표시
        combined_right_patches = np.hstack([right_patches[j] for j in range(17)])
        axes[2].imshow(combined_right_patches)
        axes[2].set_title("Right Patches")
        axes[2].axis("off")

        output_path = os.path.join(output_dir, f"sample_{image_id}.png")
        plt.savefig(output_path, bbox_inches="tight", dpi=300)
        plt.close(fig)

        print(f"✅ Saved visualization to {output_path}")

# 시각화 실행
visualize_and_save_samples(dataloader, num_samples=4)
