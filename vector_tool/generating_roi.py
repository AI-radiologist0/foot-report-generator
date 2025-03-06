import json
import matplotlib.pyplot as plt
import cv2
import numpy as np
import os
import pickle  # <-- Added for Pickle support
from tqdm import tqdm

# Load the JSON data
with open('data/json/merge/output.json', 'r') as f:
    data = json.load(f)

# # Directory to store bounding box images (optional for visualization)
# output_dir = "./bounding_boxes_output/"
# if not os.path.exists(output_dir):
#     os.makedirs(output_dir)

# Directory to save final_data as Pickle
output_pkl_path = "data/pkl/output200x300.pkl"

# Helper function to create a black image
def create_black_image(size=(200, 300)):
    return np.zeros((size[1], size[0], 3), dtype=np.uint8)

# Helper function to draw bounding boxes and save them
def extract_bboxes(image, keypoints):
    height, width, _ = image.shape
    bbox_width_size = 300
    bbox_height_size = 200  
    extracted_bboxes = [np.zeros((bbox_height_size, bbox_width_size, 3), dtype=np.uint8) for _ in range(17)]

    if len(keypoints) != 51:
        return
    
    for i in range(17):  # Joint 개수 17개 고정
        x, y, score = int(keypoints[i * 3]), int(keypoints[i * 3 + 1]), keypoints[i * 3 + 2]    
        if score > 0.0:
            x_min = max(x - bbox_width_size // 2, 0)
            y_min = max(y - bbox_height_size // 2, 0)
            x_max = min(x + bbox_width_size // 2, width)
            y_max = min(y + bbox_height_size // 2, height)
            
            cropped_bbox = image[y_min:y_max, x_min:x_max]
            
            if (x_max < x_min) or (y_max < y_min):
                return []
            try:
                # Check if the cropped bbox is valid (non-empty)
                if cropped_bbox is not None and cropped_bbox.size > 0:
                    resized_bbox = cv2.resize(cropped_bbox, (224, 224))
                    extracted_bboxes.append(resized_bbox)
                else:
                    print(f"⚠️ Warning: Empty cropped_bbox at ({x_min}, {y_min}) → Returning empty list.")
                    return []  # Return empty list if invalid region
            except cv2.error as e:
                print(f"⚠️ OpenCV Resize Error: {e} → Returning empty list.")
                return []  # Return empty list on error
            extracted_bboxes[i] = resized_bbox

    return extracted_bboxes

# Filter bbox_annotations with score >= 0.85
filtered_bboxes = [bbox for bbox in data['bboxes'] if bbox['score'] >= 0.85]

# Get unique image_ids from the filtered bbox_annotations
filtered_image_ids = list(set([bbox['image_id'] for bbox in filtered_bboxes]))

# Filter disease entries that have corresponding images
disease_entries = [d for d in data['diseases'] if d['image_id'] in filtered_image_ids]
print(f'file : {len(disease_entries)}')

# Final dictionary to store extracted bounding boxes
final_data = {}

# Process each disease entry
for disease in tqdm(disease_entries, desc="Process each disease entry"):
    image_id = disease['image_id']
    patient_id = disease.get('patient_id', '')
    file_path = disease.get('file_path', '')
    disease_class = disease.get('class_label', '')
    diagnosis = disease.get('diagnosis', '')

    # Load the corresponding image
    image_info = next((img for img in data['images'] if img['image_id'] == image_id), None)
    if image_info is None:
        continue

    image_path = image_info['file_path']
    image = cv2.imread(image_path)
    if image is None:
        print(f"Could not read image: {image_path}")
        continue

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Get keypoints and extract bounding boxes
    keypoint_annotations = [kp for kp in data['keypoints'] if kp['image_id'] == image_id]
    all_bbx_images = []
    flag = True
    for keypoint in keypoint_annotations:
        keypoints = keypoint['keypoints']
        extracted_bbx = extract_bboxes(image, keypoints)
        if extracted_bbx:
            all_bbx_images.extend(extracted_bbx)
        else:
            flag = False

    # Ensure 34 patches
    while len(all_bbx_images) < 34:
        all_bbx_images.append(create_black_image())

    all_bbx_images = all_bbx_images[:34]  # Trim if more than 34

    # Store in the final dictionary without serialization (pickle handles numpy arrays)
    if flag:
        final_data[image_id] = {
            "image_id": image_id,
            "patient_id": patient_id,
            "file_path": file_path,
            "class": disease_class,
            "diagnosis": diagnosis,
            "bbx": all_bbx_images  # Numpy arrays can be directly pickled
        }
        
print(f'After getting patches {len(final_data)}')

'''
    # Save bounding box images for visualization (Optional)
    for idx_bbx, bbox_img in enumerate(all_bbx_images):
        save_path = os.path.join(output_dir, f"{image_id}_bbox_{idx_bbx+1}.png")
        cv2.imwrite(save_path, cv2.cvtColor(bbox_img, cv2.COLOR_RGB2BGR))
'''
# # --------- Plot only the first 5 images ---------
# num_images_to_plot = min(5, len(disease_entries))
# fig, axes = plt.subplots(num_images_to_plot, 1, figsize=(10, 5 * num_images_to_plot))

# if num_images_to_plot == 1:
#     axes = [axes]  # Ensure axes is always a list

# # Plot only the first 5 images
# for idx, disease in enumerate(disease_entries[:5]):  # Only first 5 for plotting
#     image_id = disease['image_id']
#     image_info = next((img for img in data['images'] if img['image_id'] == image_id), None)
#     if image_info is None:
#         continue

#     image_path = image_info['file_path']
#     image = cv2.imread(image_path)
#     if image is None:
#         continue

#     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

#     ax = axes[idx]
#     ax.imshow(image)
#     ax.set_title(f"Image ID: {image_id} | Patient ID: {disease.get('patient_id', '')}")
#     ax.axis('off')

#     # Draw bounding boxes
#     keypoint_annotations = [kp for kp in data['keypoints'] if kp['image_id'] == image_id]
#     for keypoint in keypoint_annotations:
#         keypoints = keypoint['keypoints']
#         for i in range(0, len(keypoints), 3):
#             x = int(keypoints[i])
#             y = int(keypoints[i + 1])
#             score = keypoints[i + 2]
#             if score > 0.0:
#                 bbox_size = 224
#                 rect = plt.Rectangle((x - bbox_size // 2, y - bbox_size // 2), bbox_size, bbox_size,
#                                      linewidth=1.5, edgecolor='g', facecolor='none')
#                 ax.add_patch(rect)

# plt.tight_layout()
# plt.show()
# plt.savefig('savefig_default.png')


# --------- Save final_data as Pickle (.pkl) ---------
with open(output_pkl_path, 'wb') as pkl_file:
    pickle.dump(final_data, pkl_file)

# print(f"Bounding boxes saved to {output_dir}")
print(f"Final data saved to {output_pkl_path}")
