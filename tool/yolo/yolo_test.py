# -----------------------------------------------------------
# 
# written by Jeongmin Kim (jm.kim@dankook.ac.kr)
#
# -----------------------------------------------------------
import torch
from PIL import Image
import matplotlib.pyplot as plt
from ultralytics import YOLO
import os

# 내 훈련 모델 경로
model_path = 'ckpt/yolo/letter_foot/best.pt'  # <- 여기에 본인 pt 경로 입력

# 내 모델 로드
model = YOLO(model_path)

# 테스트할 이미지 경로
# image_path = './data/foot/gout/CAUNURI/CAUHGOUT1003_20221201202749_CR/1.2.276.0.7230010.3.1.4.67515890.5552.1670892124.186419.jpg'
image_path = 'data/RA_hand/CAUHRA10001/1.2.276.0.7230010.3.1.4.67515890.10984.1691711524.254294.jpg'

# 추론
results = model(image_path)

os.makedirs('output', exist_ok=True)
for idx, r in enumerate(results):
    img_with_boxes = r.plot()  # numpy array with bbox 시각화
    out_path = f'output/result_{idx}.jpg'
    Image.fromarray(img_with_boxes).save(out_path)