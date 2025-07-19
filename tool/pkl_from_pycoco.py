import pickle
from torch.utils.data import DataLoader

import _init_path
from dataset.cocomedical import COCOMedicalDataset  # 이전에 만든 Dataset 클래스

# 데이터셋 로드
json_path = "data/json/merge/output.json"
dataset = COCOMedicalDataset(json_path, bbox_size=120)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

output_pkl_path = "data/pkl/output.pkl"
final_data = {}

print("🔄 Extracting patches and saving to PKL...")
for sample in dataloader:
    if sample is None:
        continue
    
    image_id = sample["image_id"][0].item()
    left_patches = sample["left_patches"][0].numpy()
    right_patches = sample["right_patches"][0].numpy()
    left_keypoints = sample["left_keypoints"]
    right_keypoints = sample["right_keypoints"]

    final_data[image_id] = {
        "image_id": image_id,
        "left_patches": left_patches,
        "right_patches": right_patches,
        "left_keypoints": left_keypoints,
        "right_keypoints": right_keypoints
    }

# Save final data to pickle
with open(output_pkl_path, 'wb') as pkl_file:
    pickle.dump(final_data, pkl_file)

print(f"✅ Final data saved to {output_pkl_path}")
