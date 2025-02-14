import os
import json
import pickle
import torch
import cv2
import numpy as np
from collections import OrderedDict
import torchvision.transforms as transforms
from ultralytics import YOLO  # YOLOv5 used for detection
import _init_path
import models
from core.config import config, update_config
from core.inference import get_final_preds

PATCH_SIZE = 50
IMAGE_SIZE = (640, 360)  # Resize images for processing
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load YOLO model (ensure a model trained to detect left/right feet)
YOLO_MODEL_PATH = "ckpt/yolo/best.pt"
yolo_model = YOLO(YOLO_MODEL_PATH).to(DEVICE)

# Load keypoint detector model
def load_keypoint_detector(config_file, model_path):
    update_config(config_file)
    model = eval('models.' + config.MODEL.NAME + '.get_pose_net')(config, is_train=False)
    checkpoint = torch.load(model_path, map_location=DEVICE)
    state_dict = checkpoint.get('state_dict', checkpoint)
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        new_key = k.replace("module.", "")
        new_state_dict[new_key] = v
    model.load_state_dict(new_state_dict)
    model = model.to(DEVICE)
    model.eval()
    return model

# Generate a list of images from JSON metadata
def generate_image_list_from_json_files(json_file):
    with open(json_file, "r") as f:
        data = json.load(f)

    image_list = []
    for record in data["data"]:
        original_file_path = record["file_path"]
        if isinstance(original_file_path, list):
            continue  # Skip invalid entries

        parent_dir = os.path.dirname(original_file_path)
        patient_id = record["patient_id"]
        diagnosis_class = record["class"]
        diagnosis = record["diagnosis"]

        for img_file in os.listdir(parent_dir):
            if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                file_path = os.path.join(parent_dir, img_file)
                image_list.append({
                    "file_path": file_path,
                    "patient_id": patient_id,
                    "class": diagnosis_class,
                    "diagnosis": diagnosis
                })

    return image_list

# Detect left and right feet using YOLO
def detect_and_crop_feet(image):
    results = yolo_model(image)
    feet = {}

    for result in results:
        for bbox, cls in zip(result.boxes.xyxy, result.boxes.cls):
            x1, y1, x2, y2 = map(int, bbox)
            foot_crop = image[y1:y2, x1:x2]

            if int(cls) == 0:  # Left foot
                feet["left"] = foot_crop
            elif int(cls) == 1:  # Right foot
                feet["right"] = foot_crop

    return feet

# Preprocess image for keypoint model
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    resized_image = cv2.resize(image, (config.MODEL.IMAGE_SIZE[0], config.MODEL.IMAGE_SIZE[1]))
    image_tensor = transform(resized_image).unsqueeze(0).to(DEVICE)
    return image_tensor

# Predict keypoints using keypoint detection model
def predict_keypoints(model, image_tensor, image_size):
    center = np.array([[image_size[1] / 2, image_size[0] / 2]], dtype=np.float32)
    scale = np.array([[image_size[1] / 200, image_size[0] / 200]], dtype=np.float32)
    with torch.no_grad():
        output = model(image_tensor)
        preds, _ = get_final_preds(config, output.cpu().numpy(), center, scale)
    return preds[0]  # Return keypoints as NumPy array

# Extract patches around detected keypoints
def crop_patches(image, keypoints, patch_size=PATCH_SIZE):
    h, w = image.shape[:2]
    patches = {}

    for idx, (x, y) in enumerate(keypoints):
        key = idx + 1  # Convert index (0-based) to dictionary key (1-based)

        x, y = int(x), int(y)

        # Define patch boundaries
        x1 = max(x - patch_size // 2, 0)
        y1 = max(y - patch_size // 2, 0)
        x2 = min(x + patch_size // 2, w)
        y2 = min(y + patch_size // 2, h)

        patches[key] = image[y1:y2, x1:x2]  # Store patch

    return patches

# Save processed data to a pickle file
def save_data_as_pickle(data, save_path="foot_data.pkl"):
    if os.path.exists(save_path):
        with open(save_path, "rb") as f:
            existing_data = pickle.load(f)
    else:
        existing_data = {}

    existing_data.update(data)  # Merge new data

    with open(save_path, "wb") as f:
        pickle.dump(existing_data, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"✅ Data saved to {save_path}")

# Process images based on JSON metadata
def main(image_folder, config_file, model_path, json_file):
    # Load keypoint detection model
    model = load_keypoint_detector(config_file, model_path)

    # Load image metadata from JSON
    all_image_list = generate_image_list_from_json_files(json_file)

    results_data = {}  # Store results for all processed images

    for image_info in all_image_list:
        file_path = image_info["file_path"]
        patient_id = image_info["patient_id"]
        diagnosis = image_info["diagnosis"]
        diagnosis_class = image_info["class"]

        image = cv2.imread(file_path)
        if image is None:
            print(f"⚠️ Warning: Cannot read image {file_path}. Skipping...")
            continue

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Step 1: Detect and crop left and right foot using YOLO
        feet = detect_and_crop_feet(image)

        image_data = {
            "meta": {
                "file_path": file_path,
                "patient_id": patient_id,
                "diagnosis": diagnosis,
                "class": diagnosis_class
            },
            "data": {}
        }

        # Step 2: Process Left Foot (if detected)
        if "left" in feet:
            left_image = feet["left"]
            left_tensor = preprocess_image(left_image)
            left_keypoints = predict_keypoints(model, left_tensor, left_image.shape[:2])
            left_patches = crop_patches(left_image, left_keypoints)

            image_data["data"]["left"] = left_image
            image_data["data"]["left_keypoints"] = left_keypoints.tolist()
            image_data["data"]["left_patches"] = left_patches

        # Step 3: Process Right Foot (if detected)
        if "right" in feet:
            right_image = feet["right"]
            right_tensor = preprocess_image(right_image)
            right_keypoints = predict_keypoints(model, right_tensor, right_image.shape[:2])
            right_patches = crop_patches(right_image, right_keypoints)

            image_data["data"]["right"] = right_image
            image_data["data"]["right_keypoints"] = right_keypoints.tolist()
            image_data["data"]["right_patches"] = right_patches

        # Store results
        results_data[file_path] = image_data

    # Step 4: Save results to pickle
    save_data_as_pickle(results_data)

# Example usage
json_file = "dataset/patient_metadata.json"
config_file = "config/384x288_d256x3_adam_lr1e-3-RHPE-Foot-N3-Doctor-noflip-cau-newformat-one-val.yaml"
model_path = "ckpt/detector/model_best.pth.tar"

main("dataset/cau_dataset", config_file, model_path, json_file)
