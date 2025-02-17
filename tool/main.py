import argparse
import os, json
import cv2
import torch
from tqdm import tqdm
from ultralytics import YOLO

import _init_path
from utils.text_extractor import get_foot_side
from core.config import update_config, config


def parse_args():
    parser = argparse.ArgumentParser(description="Foot Keypoint Detection Pipeline")
    parser.add_argument("--json_path", type=str, required=True, help="Path to JSON metadata file")
    parser.add_argument("--config_file", type=str, required=True, help="Path to keypoint detector config file")
    parser.add_argument("--yolo_path", type=str, required=True, help="Path to YOLO model checkpoint")
    return parser.parse_args()


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

def get_bbox_from_yolo(yolo_model, image_info, conf_value=0.5):
    
    image_path = image_info["file_path"]
    image = cv2.imread(image_path)
    results = yolo_model(image, conf=conf_value)
    
    letter_bbox, foot_bbox = None, []
    foot_bbox_score_list = []
    
    for result in results:
        for box in result.boxes:
            x, y, w, h = box.xywh[0]
            conf = box.conf[0].item()
            class_id = int(box.cls[0].item())

            if class_id == 0:
                if conf >= 0.6:
                    letter_bbox = [x,y,w,h]
            
            elif class_id == 1:
                if conf >= 0.8:
                    foot_bbox.append([x,y,w,h])
                    foot_bbox_score_list.append(conf)

    
    return image, letter_bbox, foot_bbox, foot_bbox_score_list


def align_bbox_with_side(format, info, side, foot_bbox_list, score):
    
    format["patient_id"] = info["patient_id"]
    format["file_path"] = info["file_path"]
    score = sum(score) / len(score)
    format["score"] = score    
    if len(foot_bbox_list) == 1:
        format['side'] = side
        format["bbox"] = foot_bbox_list[0]
        
        return [format]
    
    # IF Detect two bbox.    
    if side == "right":
        flag = True
    
    sorted_bbox = sorted(foot_bbox_list, lambda bbox : bbox[0], reserve=flag)
    tmp_bucket = []
    
    for idx, bbox in enumerate(sorted_bbox):
        format["bbox"] = bbox
        if flag:
            if idx == 0:
                format['side'] = "right"
            else:
                format['side'] = "left"
        else:
            if idx == 0:
                format['side'] = "left"
            else:
                format['side'] = "right"
        
        tmp_bucket.append(format)

    return tmp_bucket
    
def main():
    args = parse_args()
    
    json_file = args.json_path
    config = args.config_file
    yolo_pt = args.yolo_path
    
    # to set up the param for joint_detector
    update_config(config)
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # declare json bucket to save
    save_json = []
        
    # Generate image_list from json(image info)
    image_list = generate_image_list_from_json(json_file)
    
    # YOLO model
    yolo_model = YOLO(yolo_pt).to(DEVICE)
    
    for image_info in tqdm(image_list, desc="Processing images"):
        
        format = {
            "file_path" : None,
            "patient_id" : None,
            "side" : None,
            "bbox" : None,
            "score" : None,
        }
        
        # First, Get the bbox list from YOLO and then save bbox list
        image, letter_bbox, foot_bbox, foot_bbox_score_list = get_bbox_from_yolo(yolo_model, image_info, conf_value=0.5)
        
        if letter_bbox is None or len(foot_bbox) == 0:
            continue # pass this image
        # Get foot side
        side = get_foot_side(image, letter_bbox)
        
        save_json.extend(align_bbox_with_side(format, image_info, side, foot_bbox, foot_bbox_score_list))
        print(letter_bbox, foot_bbox, side)
        break
    
    
    # # Save json for using with joint detector (Simple baseline) and then use val_dataset using saved json.
    # with open("/home/jmkim/foot-report-generator/data/json/joint/bbox_from_yolo.json") as f:
    #     json.dump(save_json, f)
    
    # Below step, I will implement code in another file.
    # Second, Using Dataloader for inference (joint detector)
    
    # Finally, In format below, Save pickle files.


if __name__ == "__main__":
    main()