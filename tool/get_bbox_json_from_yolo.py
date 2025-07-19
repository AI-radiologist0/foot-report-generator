import argparse
from collections import defaultdict
import os, json
import cv2
import torch
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO
from multiprocessing import Pool, cpu_count


def get_patient_id(file_path: str) -> str:
    parts = file_path.split(os.sep)
    return "_".join(parts[-3:-1])


def parse_args():
    parser = argparse.ArgumentParser(description="Foot Keypoint Detection Pipeline")
    parser.add_argument(
        "--json_path",
        type=str,
        required=False,
        default="data/json/foot_merge.json",
        help="Path to JSON metadata file (default: data/json/foot_merge.json)"
    )
    parser.add_argument(
        "--yolo_path",
        type=str,
        required=False,
        default="ckpt/yolo/letter_foot/best.pt",
        help="Path to YOLO model checkpoint (default: ckpt/yolo/letter_foot/best.pt)"
    )
    parser.add_argument(
        "--output_json_path",
        type=str,
        required=False,
        default="data/json/tmp0418/joint/bbox_from_yolo_new.json",
        help="Path to save output bbox json file (default: data/json/tmp0418/joint/bbox_from_yolo_new.json)"
    )
    return parser.parse_args()

# 이미지 로딩을 병렬 처리할 함수
def load_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        return None  # 로드 실패한 경우 None 반환
    return {"file_path": image_path, "image": image}

# Generate patient-based image list from JSON metadata (병렬 이미지 로딩 포함)
def generate_patient_image_list(json_file):
    with open(json_file, "r") as f:
        data = json.load(f)

    patient_dict = defaultdict(list)  # 딕셔너리 초기화
    image_paths = []  # 모든 이미지 경로 저장

    for record in data["data"]:
        file_paths = record["file_path"]
        if isinstance(file_paths, list):
            if not file_paths:
                continue  # 빈 리스트면 건너뜀
            for file_path in file_paths:
                if not file_path:
                    continue  # 빈 문자열도 건너뜀
                patient_id = get_patient_id(file_path)
                parent_dir = os.path.dirname(file_path)
                if patient_id not in patient_dict:
                    patient_dict[patient_id] = []
                for img_file in os.listdir(parent_dir):
                    if img_file.lower().endswith((".png", ".jpg", ".jpeg")):
                        image_paths.append((patient_id, os.path.join(parent_dir, img_file)))
        else:
            file_path = file_paths
            if not file_path:
                continue  # 빈 문자열이면 건너뜀
            patient_id = get_patient_id(file_path)
            parent_dir = os.path.dirname(file_path)
            if patient_id not in patient_dict:
                patient_dict[patient_id] = []
            for img_file in os.listdir(parent_dir):
                if img_file.lower().endswith((".png", ".jpg", ".jpeg")):
                    image_paths.append((patient_id, os.path.join(parent_dir, img_file)))

    # 병렬로 이미지 로드
    with Pool(processes=cpu_count() // 2) as pool:  # CPU 코어 절반 사용
        loaded_images = list(tqdm(pool.imap(load_image, [img[1] for img in image_paths]), 
                                  total=len(image_paths), desc="Loading Images"))

    # 로드된 이미지를 patient_dict에 정리
    for (patient_id, image_path), image_data in zip(image_paths, loaded_images):
        if image_data is not None:  # None인 경우 제외
            patient_dict[patient_id].append(image_data)

    return patient_dict

def get_batch_bbox_from_yolo(yolo_model, batch_data, patient_id):
    batch_images = [data["image"] for data in batch_data]  # 로드된 이미지 리스트 전달
    results = yolo_model(batch_images, conf=0.589999999)
    batch_output = []
    
    for img_idx, result in enumerate(results):
        letter_bbox = None
        foot_bboxes = []
        foot_scores = []
        img_width, img_height = batch_data[img_idx]["image"].shape[1], batch_data[img_idx]["image"].shape[0]
        
        for box in result.boxes:
            x_center, y_center, w, h = map(float, box.xywh[0].cpu().numpy())  # Ensure float format for JSON
            conf = float(box.conf[0].item())  # Convert to standard float
            class_id = int(box.cls[0].item())

            # 변환: (x_center, y_center, w, h) → (x_min, y_min, w, h)
            x_min = max(0, x_center - (w / 2))
            y_min = max(0, y_center - (h / 2))
            w = min(w, img_width - x_min)
            h = min(h, img_height - y_min)

            if class_id == 0 and conf >= 0.6:  # Letter detection
                letter_bbox = [x_min, y_min, w, h]
            elif class_id == 1 and conf >= 0.8:  # Foot detection
                foot_bboxes.append([x_min, y_min, w, h])
                foot_scores.append(conf)
        
        if letter_bbox is None and not foot_bboxes:
            continue  # Skip images with no useful detections
        
        for bbox, score in zip(foot_bboxes, foot_scores):
            batch_output.append({
                "image_id": 0, # garbage image_id,
                "patient_id": patient_id,
                "file_path": batch_data[img_idx]["file_path"],  # 원본 파일 경로 유지
                "letter_bbox": letter_bbox,
                "bbox": bbox,
                "category_id": 1,
                "score": score
            })
    
    return batch_output

def main():
    args = parse_args()
    json_file = args.json_path
    yolo_pt = args.yolo_path
    output_json_path = args.output_json_path
    
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("🔄 Starting parallel image loading...")
    patient_dict = generate_patient_image_list(json_file)
    print("✅ Image loading complete!")

    yolo_model = YOLO(yolo_pt).to(DEVICE)
    
    save_json = []
    for patient_id, batch_data in tqdm(patient_dict.items(), desc=f"Processing Patients"):
        print(f"Processing patient: {patient_id}")
        batch_results = get_batch_bbox_from_yolo(yolo_model, batch_data, patient_id=patient_id)
        save_json.extend(batch_results)
    
    file_paths = sorted(set(item["file_path"] for item in save_json))
    file_path_to_id = {fp: i + 1 for i, fp in enumerate(file_paths)}

    for item in save_json:
        item["image_id"] = file_path_to_id[item["file_path"]]

    # output_json_path로 지정된 폴더가 없으면 생성
    output_dir = os.path.dirname(output_json_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    with open(output_json_path, "w") as f:
        json.dump(save_json, f)
    
if __name__ == "__main__":
    main()
