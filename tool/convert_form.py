import json
import os

def convert_bbox_json(input_json_path, output_json_path):
    with open(input_json_path, 'r') as f:
        data = json.load(f)
    
    images = []
    annotations = []
    image_id_map = {}
    image_id_counter = 1
    
    for entry in data:
        file_path = entry["file_path"]
        
        if file_path not in image_id_map:
            image_id_map[file_path] = image_id_counter
            images.append({
                "image_id": image_id_counter,
                "file_path": file_path,
            })
            image_id_counter += 1
        
        entry["image_id"] = image_id_map[file_path]
        annotations.append(entry)
    
    output_data = {
        "images": images,
        "annotations": annotations
    }
    
    with open(output_json_path, 'w') as f:
        json.dump(output_data, f, indent=4)
    
    print(f"Converted JSON saved to {output_json_path}")

# Example usage
if __name__ == "__main__":
    input_json_path = "data/json/joint/ocr_results_v1.json"  # Input file
    output_json_path = "data/json/joint/ocr_results_v1.1.json"  # Output file
    convert_bbox_json(input_json_path, output_json_path)
