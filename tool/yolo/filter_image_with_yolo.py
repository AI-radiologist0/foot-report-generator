# -----------------------------------------------------------
# 
# written by Jeongmin Kim (jm.kim@dankook.ac.kr)
#
# -----------------------------------------------------------
import json
import os
import argparse
from ultralytics import YOLO
import cv2
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

# Argument Parser 설정
def parse_args():
    parser = argparse.ArgumentParser(description="Foot Region Detector using YOLO")
    parser.add_argument("--json_file", type=str, required=True, help="Path to the input JSON file containing file paths")
    parser.add_argument("--output_json", type=str, required=True, help="Path to save the output JSON file")
    parser.add_argument("--yolo_model", type=str, required=True, help="Path to the YOLO model file")
    parser.add_argument("--conf_thresh", type=float, default=0.9, help="Confidence threshold for YOLO detection")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for YOLO inference")
    return parser.parse_args()

# JSON 데이터 생성 함수 (배치 처리)
def generate_vgg_input_json(image_list, yolo_model, conf_thresh, batch_size, output_json):
    output_data = {"data": []}

    # 이미지를 배치로 나누기
    for i in tqdm(range(0, len(image_list), batch_size), desc="Processing batches"):
        batch = image_list[i:i + batch_size]
        batch_images = []
        batch_records = []

        # 배치 구성
        for record in batch:
            file_path = record["file_path"]
            if os.path.exists(file_path):
                img = cv2.imread(file_path)
                if img is not None:
                    batch_images.append(img)
                    batch_records.append(record)

        # YOLO 추론 실행
        if batch_images:
            results = yolo_model(batch_images, conf=conf_thresh)

            # 배치 결과 처리
            for result, record in zip(results, batch_records):
                has_valid_box = False
                valid_boxes = []
                valid_box_conf = 0.0
                is_zero = 1

                for box in result.boxes:
                    valid_box_conf = box.conf.item()
                    if valid_box_conf > conf_thresh:
                        has_valid_box = True
                        bbox = box.xyxy.tolist()[0]
                        valid_boxes.append({
                            "bbox": bbox,
                            "confidence": valid_box_conf
                        })
                        break

                # 조건에 맞는 이미지만 JSON에 추가
                if has_valid_box:
                    output_data["data"].append({
                        "file_path": record["file_path"],
                        "patient_id": record["patient_id"],
                        "class": record["class"],
                        "valid_boxes": valid_boxes
                    })

    # JSON 파일 저장
    with open(output_json, "w") as f:
        json.dump(output_data, f, indent=4)

# 메인 실행부
def main():
    args = parse_args()

    # JSON 파일 읽기
    with open(args.json_file, 'r') as f:
        data = json.load(f)

    # 이미지 리스트 생성
    image_list = []
    for record in data["data"]:
        original_file_path = record["file_path"]
        if isinstance(original_file_path, list):
            continue
        parent_dir = os.path.dirname(original_file_path)
        patient_id = record["patient_id"]
        diagnosis_class = record["class"]

        for img_file in os.listdir(parent_dir):
            if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                file_path = os.path.join(parent_dir, img_file)
                image_list.append({
                    "file_path": file_path,
                    "patient_id": patient_id,
                    "class": diagnosis_class
                })

    # YOLO 모델 로드
    yolo_model = YOLO(args.yolo_model)

    # VGG 입력용 JSON 생성
    generate_vgg_input_json(image_list, yolo_model, args.conf_thresh, args.batch_size, args.output_json)

if __name__ == "__main__":
    main()
