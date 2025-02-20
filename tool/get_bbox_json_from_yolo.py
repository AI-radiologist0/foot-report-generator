import argparse
import os, json
import cv2
import torch
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO
from multiprocessing import Pool, cpu_count


def parse_args():
    parser = argparse.ArgumentParser(description="Foot Keypoint Detection Pipeline")
    parser.add_argument("--json_path", type=str, required=True, help="Path to JSON metadata file")
    parser.add_argument("--yolo_path", type=str, required=True, help="Path to YOLO model checkpoint")
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

    patient_dict = {}
    image_paths = []  # 모든 이미지 경로 저장

    for record in data["data"]:
        patient_id = record["patient_id"]
        parent_dir = os.path.dirname(record["file_path"])
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

def get_batch_bbox_from_yolo(yolo_model, batch_data):
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
    
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("🔄 Starting parallel image loading...")
    patient_dict = generate_patient_image_list(json_file)
    print("✅ Image loading complete!")

    yolo_model = YOLO(yolo_pt).to(DEVICE)
    
    save_json = []
    for patient_id, batch_data in tqdm(patient_dict.items(), desc="Processing Patients"):
        batch_results = get_batch_bbox_from_yolo(yolo_model, batch_data)
        save_json.extend(batch_results)
    
    output_json_path = "/home/jmkim/foot-report-generator/data/json/joint/bbox_from_yolo.json"
    with open(output_json_path, "w") as f:
        json.dump(save_json, f)
    
if __name__ == "__main__":
    main()
