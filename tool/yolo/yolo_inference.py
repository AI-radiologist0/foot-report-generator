# -----------------------------------------------------------
# YOLOv5 Training Script using Ultralytics
# written by Jeongmin Kim (jm.kim@dankook.ac.kr)
# -----------------------------------------------------------
import argparse
from ultralytics import YOLO
from config import cfg, update_config  # Config 로드 함수
import models  # YOLO 모델 생성 함수 포함
import os

def main():
    # ArgumentParser로 학습 설정 받기
    parser = argparse.ArgumentParser(description='Train YOLOv5 model for X-ray foot detection')
    parser.add_argument('--cfg', type=str, required=True, help='Path to config file')
    parser.add_argument('--input_dir', type=str, required=True, help='Path to input image file.')
    args = parser.parse_args()

    # Config 파일 불러오기 및 업데이트
    update_config(cfg, args)  # 명령줄 인자로 받은 값을 반영하여 Config 업데이트
    cfg.freeze()

    # 모델 생성 (YOLOv5) - inference 용 모델델
    model = models.create('YOLOV5', cfg)

    print("Running inference...")
    results = model.predict(
        source=args.input_dir,  # 검증 이미지 경로
        save=True,                    # 결과 저장 여부        
    )


if __name__ == '__main__':
    main()