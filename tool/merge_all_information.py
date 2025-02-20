import json
import os
from tqdm import tqdm  # Library for displaying progress bars
from collections import defaultdict

def load_json(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)

def build_lookup(data, key_func):
    lookup = {}
    for item in data:
        key = key_func(item["file_path"])  # Extract 'file_path' first before applying function
        lookup[key] = item
    return lookup

def get_folder_path(file_path):
    return os.path.dirname(file_path)  # Extracts the folder path

def merge_keypoints_with_annotations(bbox_json_path, keypoints_json_path, disease_json_path, output_json_path):
    bbox_data = load_json(bbox_json_path)  # Bounding box data
    keypoints_data = load_json(keypoints_json_path)  # Keypoints data
    disease_data = load_json(disease_json_path)  # Disease data
    
    image_id_map = {}
    image_id_counter = 1
    
    # Build dictionary lookup based on folder path instead of file path
    disease_lookup = build_lookup(disease_data["data"], get_folder_path)
    bbox_annotations = defaultdict(list)
    keypoint_annotations = defaultdict(list)
    
    for image in tqdm(bbox_data["images"], desc="Processing Images"):
        file_name = image["image_id"]
        if file_name not in image_id_map:
            image_id_map[file_name] = image_id_counter
            image["image_id"] = image_id_counter
            image_id_counter += 1
    
    bbox_annotation_id_counter = 1
    for ann in tqdm(bbox_data["annotations"], desc="Processing BBox Annotations"):
        image_id = ann["image_id"]
        if image_id in image_id_map:
            ann["image_id"] = image_id_map[image_id]
            ann["bbox_annotation_id"] = bbox_annotation_id_counter
            bbox_annotation_id_counter += 1
            if "detected_text" in ann and ann["detected_text"] in ["R", "L"]:
                bbox_annotations[image_id].append(ann)
    
    keypoint_annotation_id_counter = 1
    for keypoint in tqdm(keypoints_data, desc="Processing Keypoints Annotations"):
        original_id = keypoint["image_id"]
        if original_id in image_id_map:
            keypoint["image_id"] = image_id_map[original_id]
            keypoint["keypoint_annotation_id"] = keypoint_annotation_id_counter
            keypoint_annotation_id_counter += 1
            keypoint_annotations[image_id_map[original_id]].append(keypoint)
    
    disease_list = []
    for image in tqdm(bbox_data["images"], desc="Processing Disease Data in Image Order"):
        image_id = image["image_id"]
        file_path = image["file_path"]
        folder_path = get_folder_path(file_path)
        disease_entry = disease_lookup.get(folder_path, {})
        
        patient_id = disease_entry.get("patient_id")
        disease_class = disease_entry.get("class")
        diagnosis = disease_entry.get("diagnosis")
        
        associated_bbox_annotations = [ann["bbox_annotation_id"] for ann in bbox_annotations[image_id]]
        sorted_k_ann = sorted(keypoint_annotations[image_id], key=lambda k: k["center"][0])
        
        left_keypoint, right_keypoint = [], []
        f_ann = sorted(bbox_annotations[image_id], key=lambda b: b["bbox"][0])
        side = next((b["detected_text"] for b in bbox_annotations[image_id]), "")
        
        if len(f_ann) == 1:
            if side == "R":
                right_keypoint = sorted_k_ann
            elif side == "L":
                left_keypoint = sorted_k_ann
            else:
                continue
        elif len(f_ann) == 2:
            if side == "R":
                right_keypoint = sorted_k_ann[:len(sorted_k_ann)//2]
                left_keypoint = sorted_k_ann[len(sorted_k_ann)//2:]
            elif side == "L":
                left_keypoint = sorted_k_ann[:len(sorted_k_ann)//2]
                right_keypoint = sorted_k_ann[len(sorted_k_ann)//2:]
            else:
                continue
        else:
            continue
        
        disease_list.append({
            "image_id": image_id,
            "patient_id": patient_id,
            "file_path": file_path,
            "class": disease_class,
            "diagnosis": diagnosis,
            "bbox_annotation_ids": associated_bbox_annotations,
            "keypoint_annotation_ids": {"left": left_keypoint, "right": right_keypoint}
        })
    
    output_data = {
        "images": bbox_data["images"],
        "bbox_annotations": list(bbox_annotations.values()),
        "keypoint_annotations": list(keypoint_annotations.values()),
        "disease": disease_list,
        "meta": disease_data.get("meta", {})
    }
    
    if not os.path.exists(os.path.dirname(output_json_path)):
        os.makedirs(os.path.dirname(output_json_path))
    
    with open(output_json_path, 'w') as f:
        json.dump(output_data, f, indent=4)
    
    print(f"Merged JSON saved to {output_json_path}")

if __name__ == "__main__":
    bbox_json_path = "data/json/joint/ocr_results_new.json"
    keypoints_json_path = "output/coco/pose_resnet_50/384x288_d256x3_adam_lr1e-3-RHPE-Foot-N3-Doctor-noflip-cau-newformat-one-val/results/keypoints_JPEGImages_results.json"
    disease_json_path = "data/json/foot_merge.json"
    output_json_path = "data/json/merge/final_merge_output.json"
    merge_keypoints_with_annotations(bbox_json_path, keypoints_json_path, disease_json_path, output_json_path)
