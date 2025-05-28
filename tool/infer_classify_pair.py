import os
import json
import torch
import pandas as pd
from PIL import Image
from torchvision import models, transforms
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# -----------------------------
# 설정
# -----------------------------
json_path = "data/json/tmp0418/joint/ocr_results_v3.json"
model_path = "resnet18_best_include2.pth"
output_json = "data/json/tmp0418/joint/ocr_results_pairing3.json"
batch_size = 64
num_workers = 8
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = 3

# -----------------------------
# 전처리 정의
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

# -----------------------------
# JSON 로드 & 유일 이미지 추출
# -----------------------------
with open(json_path, 'r') as f:
    raw_data = json.load(f)

unique_paths = {}
for item in raw_data:
    path = item["file_path"]
    image_id = item.get("image_id", -1)
    unique_paths[path] = image_id

file_paths = list(unique_paths.keys())
image_ids = [unique_paths[p] for p in file_paths]

# -----------------------------
# 병렬 이미지 로딩 함수
# -----------------------------
def load_single_image(args):
    path, image_id = args
    try:
        img = Image.open(path).convert("RGB")
        img_tensor = transform(img)
        return (img_tensor, path, image_id)
    except Exception as e:
        print(f"❌ {path} 로딩 실패: {e}")
        return None

def preload_images_parallel(paths, ids, max_workers=8):
    images, valid_paths, valid_ids = [], [], []
    args_list = list(zip(paths, ids))

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(load_single_image, args): args for args in args_list}

        for future in tqdm(as_completed(futures), total=len(futures), desc="🔄 Preloading (Parallel)"):
            result = future.result()
            if result is not None:
                img_tensor, path, image_id = result
                images.append(img_tensor)
                valid_paths.append(path)
                valid_ids.append(image_id)

    return images, valid_paths, valid_ids

# -----------------------------
# 병렬 이미지 전처리
# -----------------------------
loaded_images, valid_paths, valid_ids = preload_images_parallel(file_paths, image_ids, max_workers=num_workers)

# -----------------------------
# 모델 로딩
# -----------------------------
model = models.resnet18(pretrained=False)
model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
model.load_state_dict(torch.load(model_path, map_location=device))
model = model.to(device)
model.eval()

# -----------------------------
# 추론
# -----------------------------
results = []
all_images = torch.stack(loaded_images)
total_batches = (len(all_images) + batch_size - 1) // batch_size

with torch.no_grad():
    for i in tqdm(range(total_batches), desc="🚀 Inference"):
        batch = all_images[i * batch_size: (i + 1) * batch_size].to(device)
        outputs = model(batch)
        preds = torch.argmax(outputs, dim=1).cpu().tolist()

        for j, pred in enumerate(preds):
            idx = i * batch_size + j
            results.append({
                "image_id": valid_ids[idx],
                "file_path": valid_paths[idx],
                "predicted_class": pred
            })

# -----------------------------
# 결과를 기존 JSON에 병합
# -----------------------------
pred_map = {(item["file_path"], item["image_id"]): item["predicted_class"] for item in results}

for item in raw_data:
    key = (item["file_path"], item.get("image_id", -1))
    if key in pred_map:
        item["predicted_class"] = pred_map[key]

# -----------------------------
# JSON 저장
# -----------------------------

raw_data = sorted(raw_data, key=lambda x:x["image_id"])

with open(output_json, "w") as f:
    json.dump(raw_data, f, indent=4)

print(f"✅ 완료: {output_json} 저장됨")
