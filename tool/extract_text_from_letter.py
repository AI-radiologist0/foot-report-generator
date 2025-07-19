import json
import cv2
import torch
import re
import easyocr
import numpy as np
from torchvision import transforms
from tqdm import tqdm
import argparse
import os

# GPU 사용 여부 확인
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# EasyOCR 모델 로드 (GPU 사용)
reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available())

def parse_args():
    parser = argparse.ArgumentParser(description="Extract text from letter bbox using OCR")
    parser.add_argument('--json_path', type=str, required=False, default="/home/jmkim/foot-report-generator/data/json/tmp0418/joint/bbox_from_yolo_v3.json", help="Input bbox json file (default: 발 X-ray 기준)")
    parser.add_argument('--output_path', type=str, required=False, default="/home/jmkim/foot-report-generator/data/json/tmp0418/joint/ocr_results_v3.json", help="Output OCR json file (default: 발 X-ray 기준)")
    return parser.parse_args()

def load_json(json_path):
    """JSON 파일을 로드하고 데이터 리스트 반환"""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def process_image(file_path, letter_bbox, target_size=(200, 200)):
    """letter_bbox 영역을 자르고 GRAY 이미지로 변환"""
    image = cv2.imread(file_path)
    if image is None:
        raise ValueError(f"Image at {file_path} could not be loaded.")

    x_min, y_min, w, h = map(int, letter_bbox)
    roi = image[y_min:y_min+h, x_min:x_min+w]
    resized_roi = cv2.resize(roi, target_size, interpolation=cv2.INTER_AREA)
    gray_roi = cv2.cvtColor(resized_roi, cv2.COLOR_BGR2GRAY)
    return gray_roi

def extract_text(image) -> str:
    """EasyOCR을 통해 텍스트 추출 후 L/R만 반환"""
    results = reader.readtext(image)
    extracted_text = []

    for _, text, _ in results:
        text = re.sub(r'[^A-Z]', '', text.strip().upper())
        if text in ["R", "RST"]:
            extracted_text.append("R")
        elif text in ["L", "LST"]:
            extracted_text.append("L")

    return " ".join(extracted_text)

def process_json(json_path):
    """JSON의 각 항목에 대해 OCR 수행 후 결과 저장"""
    data = load_json(json_path)
    results = []
    count = 0

    with tqdm(total=len(data), desc="Processing Images", unit="img") as pbar:
        for item in data:
            file_path = item["file_path"]
            letter_bbox = item["letter_bbox"]

            detected_text = ""
            try:
                roi = process_image(file_path, letter_bbox)
                detected_text = extract_text(roi) or ""
                tqdm.write(f"✅ Processed: {file_path} -> Text: {detected_text}")
                count += 1
            except Exception as e:
                tqdm.write(f"❌ Error processing {file_path}: {str(e)}")

            results.append({
                "file_path": file_path,
                "letter_bbox": letter_bbox,
                "bbox": item['bbox'],
                "patient_id": item["patient_id"],
                "image_id": item["image_id"],
                "category_id": item["category_id"],
                "score": item["score"],
                "detected_text": detected_text,
            })

            pbar.update(1)

    print(f"After Processing remain images : {count}")
    return results

if __name__ == "__main__":
    args = parse_args()
    json_path = args.json_path
    output_path = args.output_path

    ocr_results = process_json(json_path)

    # output_path의 상위 폴더가 없으면 생성
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(ocr_results, f, ensure_ascii=False, indent=4)

    print(f"\n🎯 OCR 결과 저장 완료: {output_path}")
    print("The number of OCR results: ", len(ocr_results))
