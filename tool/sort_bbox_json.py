import json

with open('data/json/tmp0418/joint/ocr_results_v2.json', 'r') as f:
    keypoint_data = json.load(f)


keypoint_data = sorted(keypoint_data, key=lambda x:x['image_id'])

with open("data/json/tmp0418/joint/ocr_results_v2_sorted.json", "w") as f:
    json.dump(keypoint_data, f, indent=4)