import os
import json
import pickle
import argparse
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description="Generate Pickle Dump for Foot Keypoint Dataset")
    parser.add_argument("--json_path", type=str, required=True, help="Path to JSON metadata file")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save output pickle file")
    return parser.parse_args()

def convert_bbox_format(bbox):
    """Convert bbox format from (x, y, w, h) to (x1, y1, x2, y2)"""
    x, y, w, h = bbox
    x1, y1 = x, y
    x2, y2 = x + w, y + h
    return [x1, y1, x2, y2]

def generate_pickle(json_path, output_path):
    with open(json_path, "r") as f:
        data = json.load(f)
    
    formatted_data = {}
    for idx, item in tqdm(enumerate(data), total=len(data), desc="Processing data"):
        meta_info = {
            "file_path": item["file_path"],
            "patient_id": item.get("patient_id", "Unknown"),
            "side": item.get("side", "right"),
            "diagnosis": item.get("diagnosis", "degenerative change"),
            "class": item.get("class", "OA"),
            "resized_size": {"width": 640, "height": 640}  # Placeholder, update if needed
        }
        
        bbox = convert_bbox_format(item["foot_bbox"]) if "foot_bbox" in item else None
        keypoints = item.get("keypoints", [])
        pathes = item.get("pathes", {})
        
        formatted_data[idx] = {
            "meta": meta_info,
            "data": {
                "bbox": bbox,
                "keypoints": keypoints,
                "pathes": pathes
            }
        }
    
    with open(output_path, "wb") as f:
        pickle.dump(formatted_data, f)
    
    print(f"✅ Pickle file saved at {output_path}")

if __name__ == "__main__":
    args = parse_args()
    generate_pickle(args.json_path, args.output_path)
