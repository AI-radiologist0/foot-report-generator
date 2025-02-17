import os
import json
import pickle
import torch
import cv2
import numpy as np
from collections import OrderedDict
import torchvision.transforms as transforms
from ultralytics import YOLO
from tqdm import tqdm  # Progress bar

import _init_path
import models
from core.config import config, update_config
from core.inference import get_final_preds

PATCH_SIZE = 50
IMAGE_SIZE = (640, 360)  # Resize images for processing
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load YOLO model
YOLO_MODEL_PATH = "ckpt/yolo/foot_lr/last.pt"
yolo_model = YOLO(YOLO_MODEL_PATH).to(DEVICE)

# Load keypoint detector model
def load_keypoint_detector(config_file, model_path):
    update_config(config_file)
    model = eval('models.' + config.MODEL.NAME + '.get_pose_net')(config, is_train=False)
    checkpoint = torch.load(model_path, map_location=DEVICE)
    state_dict = checkpoint.get('state_dict', checkpoint)
    new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict)
    model.to(DEVICE).eval()
    return model

# Generate image list from JSON metadata
def generate_image_list_from_json(json_file):
    with open(json_file, "r") as f:
        data = json.load(f)

    image_list = []
    for record in data["data"]:
        parent_dir = os.path.dirname(record["file_path"])
        for img_file in os.listdir(parent_dir):
            if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_list.append({
                    "file_path": os.path.join(parent_dir, img_file),
                    "patient_id": record["patient_id"],
                    "class": record["class"],
                    "diagnosis": record["diagnosis"]
                })

    return image_list

# Detect left and right feet using YOLO
def detect_feet(image):
    results = yolo_model(image)
    feet = {}

    for result in results:
        boxes = result.boxes.xyxy
        confs = result.boxes.conf
        classes = result.boxes.cls

        foot_candidates = {"left": [], "right": []}
        for bbox, conf, cls in zip(boxes, confs, classes):
            x1, y1, x2, y2 = map(int, bbox)
            if int(cls) == 0:
                foot_candidates["left"].append((bbox.tolist(), conf.item()))
            elif int(cls) == 1:
                foot_candidates["right"].append((bbox.tolist(), conf.item()))

        # Select the best detection (highest confidence)
        for foot_side in ["left", "right"]:
            if foot_candidates[foot_side]:
                best_bbox, _ = max(foot_candidates[foot_side], key=lambda x: x[1])
                feet[foot_side] = best_bbox

    return feet

# Preprocess image for keypoint model
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    resized_image = cv2.resize(image, (config.MODEL.IMAGE_SIZE[0], config.MODEL.IMAGE_SIZE[1])) # 384 X 288 
    return transform(resized_image).unsqueeze(0).to(DEVICE)

# Predict keypoints
def predict_keypoints(model, image_tensor, image_size):
    center = np.array([[image_size[1] / 2, image_size[0] / 2]], dtype=np.float32)
    scale = np.array([[image_size[1] / 200, image_size[0] / 200]], dtype=np.float32)
    with torch.no_grad():
        output = model(image_tensor)
        preds, _ = get_final_preds(config, output.cpu().numpy(), center, scale)
    return preds[0] if preds is not None and len(preds[0]) > 0 else []

# Extract patches around detected keypoints
def crop_patches(image, keypoints):
    h, w = image.shape[:2]
    patches_xyxy = {}

    for idx, (x, y) in enumerate(keypoints):
        key = idx + 1
        x, y = int(x), int(y)
        x1, y1 = max(x - PATCH_SIZE // 2, 0), max(y - PATCH_SIZE // 2, 0)
        x2, y2 = min(x + PATCH_SIZE // 2, w), min(y + PATCH_SIZE // 2, h)
        patches_xyxy[key] = [x1, y1, x2, y2]

    return patches_xyxy

# Save data to pickle
def save_data_as_pickle(data, save_path="foot_data.pkl"):
    if os.path.exists(save_path):
        with open(save_path, "rb") as f:
            existing_data = pickle.load(f)
    else:
        existing_data = {}

    existing_data.update(data)

    with open(save_path, "wb") as f:
        pickle.dump(existing_data, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"✅ Data saved to {save_path}")

# Main processing function with tqdm progress bar
def main(config_file, model_path, json_file):
    model = load_keypoint_detector(config_file, model_path)
    all_image_list = generate_image_list_from_json(json_file)
    results_data = {}

    for idx, image_info in tqdm(enumerate(all_image_list, start=1), total=len(all_image_list), desc="Processing Images"):
        file_path = image_info["file_path"]
        image = cv2.imread(file_path)
        if image is None:
            print(f"⚠️ Warning: Cannot read image {file_path}. Skipping...")
            continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Step 1: Detect feet bounding boxes
        feet_bboxes = detect_feet(image)

        image_data = {
            "meta": {
                "file_path": file_path,
                "patient_id": image_info["patient_id"],
                "diagnosis": image_info["diagnosis"],
                "class": image_info["class"]
            },
            "data": {}
        }

        # Step 2: Process Left Foot
        if "left" in feet_bboxes:
            x1, y1, x2, y2 = map(int, feet_bboxes["left"])
            left_cropped = image[y1:y2, x1:x2]
            left_tensor = preprocess_image(left_cropped)
            left_keypoints = predict_keypoints(model, left_tensor, left_cropped.shape[:2])
            left_patches = crop_patches(left_cropped, left_keypoints)

            image_data["data"]["left_bbox"] = [x1, y1, x2, y2]
            image_data["data"]["left_keypoints"] = left_keypoints.tolist() if len(left_keypoints) > 0 else []
            image_data["data"]["left_patches"] = left_patches

        # Step 3: Process Right Foot
        if "right" in feet_bboxes:
            x1, y1, x2, y2 = map(int, feet_bboxes["right"])
            right_cropped = image[y1:y2, x1:x2]
            right_tensor = preprocess_image(right_cropped)
            right_keypoints = predict_keypoints(model, right_tensor, right_cropped.shape[:2])
            right_patches = crop_patches(right_cropped, right_keypoints)

            image_data["data"]["right_bbox"] = [x1, y1, x2, y2]
            image_data["data"]["right_keypoints"] = right_keypoints.tolist() if len(right_keypoints) > 0 else []
            image_data["data"]["right_patches"] = right_patches

        results_data[idx] = image_data

    save_data_as_pickle(results_data, 'all_foot_data.pkl')

# Example usage
json_file = "data/json/foot_merge.json"
config_file = "config/384x288_d256x3_adam_lr1e-3-RHPE-Foot-N3-Doctor-noflip-cau-newformat-one-val.yaml"
model_path = "ckpt/detector/model_best.pth.tar"

main(config_file, model_path, json_file)
