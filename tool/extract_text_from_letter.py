import json
import cv2
import torch
import re
import easyocr
import numpy as np
from torchvision import transforms
from tqdm import tqdm

# GPU 사용 여부 확인
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# EasyOCR 모델 로드 (GPU 사용)
reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available())

def load_json(json_path):
    """JSON 파일을 로드하고 데이터 리스트 반환"""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def process_image(file_path, letter_bbox, target_size=(200, 200)):
    """
    이미지에서 letter_bbox 영역을 추출한 후, 100x100 크기로 변환하고 GPU 텐서로 변환
    """
    # 이미지 로드
    image = cv2.imread(file_path)
    if image is None:
        raise ValueError(f"Image at {file_path} could not be loaded.")

    # BBox 좌표 추출
    x_min, y_min, w, h = map(int, letter_bbox)


    # ROI (Region of Interest) 자르기
    roi = image[y_min:y_min+h, x_min:x_min+w]

    # 크기 조정 (100x100)
    resized_roi = cv2.resize(roi, target_size, interpolation=cv2.INTER_AREA)

    # EasyOCR을 위한 변환 (BGR → GRAY)
    gray_roi = cv2.cvtColor(resized_roi, cv2.COLOR_BGR2GRAY)
    # gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    return gray_roi  # EasyOCR은 grayscale 이미지를 처리 가능


def extract_text(image):
    """
    EasyOCR을 사용해 이미지 내 텍스트를 추출하며, 
    R, RST → "R", L, LST → "L" 변환
    나머지는 "None"으로 반환
    """
    results = reader.readtext(image)
    
    extracted_text = []
    for _, text, _ in results:  # OCR 결과에서 텍스트 부분만 가져오기
        text = text.strip().upper()  # 대소문자 통일

        # 불필요한 문자 삭제
        text = re.sub(r'[^A-Z]', '', text)
        
        # 특정 변환 규칙 적용
        if text in ["R", "RST"]:
            extracted_text.append("R")
        elif text in ["L", "LST"]:
            extracted_text.append("L")

    return " ".join(extracted_text)  # 리스트를 문자열로 변환



def process_json(json_path):
    """
    JSON 파일을 로드하고, 각 이미지의 letter_bbox에서 ROI를 추출하여 EasyOCR로 텍스트 인식
    """
    data = load_json(json_path)
    ann = data["annotations"]
    results = []
    count = 0
    with tqdm(total=len(ann), desc="Processing Images", unit="img") as pbar:
        for item in ann:
            file_path = item["file_path"]
            letter_bbox = item["letter_bbox"]

            try:
                roi = process_image(file_path, letter_bbox)  # ROI 추출
                text = extract_text(roi)  # EasyOCR로 텍스트 인식
                if text is None:
                    continue
                results.append({
                    "file_path": file_path,
                    "letter_bbox": letter_bbox,
                    "bbox": item['bbox'],
                    "image_id": item["image_id"],
                    "score": item["score"],
                    "detected_text": text,
                })
                count += 1
                tqdm.write(f"✅ Processed: {file_path} -> Text: {text}")
            except Exception as e:
                tqdm.write(f"❌ Error processing {file_path}: {str(e)}")
            pbar.update(1)  

    print(f"After Processing remain images : {count}")
    
    return results

# 예제 JSON 파일 경로
json_path = "data/json/joint/bbox_from_yolo_new.json"

# 실행
ocr_results = process_json(json_path)

# 결과 저장 (JSON 파일로 저장 가능)
output_path = "./data/json/joint/ocr_results_v1.json"
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(ocr_results, f, ensure_ascii=False, indent=4)

print(f"\n🎯 OCR 결과 저장 완료: {output_path}")
