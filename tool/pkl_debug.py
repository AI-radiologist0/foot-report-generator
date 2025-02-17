import pickle
import cv2
import os

# Directory to save extracted patches and visualizations
PATCH_SAVE_DIR = "extracted_patches"
ANNOTATED_SAVE_DIR = "annotated_images"
os.makedirs(PATCH_SAVE_DIR, exist_ok=True)
os.makedirs(ANNOTATED_SAVE_DIR, exist_ok=True)

# Load the pickle file
def load_pickle_file(pickle_path):
    with open(pickle_path, "rb") as f:
        data = pickle.load(f)
    return data

# Draw bounding boxes and keypoints
def visualize_annotations(image, bbox, keypoints, color=(0, 255, 0), keypoint_color=(0, 0, 255)):
    annotated_image = image.copy()

    # Draw Bounding Box
    if bbox:
        x1, y1, x2, y2 = bbox
        cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, 2)

    # Draw Keypoints
    for x, y in keypoints:
        cv2.circle(annotated_image, (int(x), int(y)), 5, keypoint_color, -1)

    return annotated_image

# Extract patches and visualize bounding boxes & keypoints
def extract_patches_and_visualize(data):
    for idx, image_data in data.items():
        file_path = image_data["meta"]["file_path"]
        original_image = cv2.imread(file_path)

        if original_image is None:
            print(f"⚠️ Warning: Could not read {file_path}. Skipping...")
            continue

        # Create a copy for annotation
        annotated_image = original_image.copy()

        # Process left foot
        if "left_bbox" in image_data["data"]:
            left_bbox = image_data["data"]["left_bbox"]
            left_keypoints = image_data["data"].get("left_keypoints", [])
            annotated_image = visualize_annotations(annotated_image, left_bbox, left_keypoints, color=(255, 0, 0))

        # Process right foot
        if "right_bbox" in image_data["data"]:
            right_bbox = image_data["data"]["right_bbox"]
            right_keypoints = image_data["data"].get("right_keypoints", [])
            annotated_image = visualize_annotations(annotated_image, right_bbox, right_keypoints, color=(0, 255, 0))

        # Save annotated image
        annotated_path = os.path.join(ANNOTATED_SAVE_DIR, f"{idx}_annotated.jpg")
        cv2.imwrite(annotated_path, annotated_image)

        # Process patches
        for foot_side in ["left", "right"]:
            if f"{foot_side}_patches" in image_data["data"]:
                patch_dir = os.path.join(PATCH_SAVE_DIR, f"{idx}_{foot_side}")
                os.makedirs(patch_dir, exist_ok=True)

                for keypoint_id, (x1, y1, x2, y2) in image_data["data"][f"{foot_side}_patches"].items():
                    patch = original_image[y1:y2, x1:x2]
                    cv2.imwrite(os.path.join(patch_dir, f"patch_{keypoint_id}.jpg"), patch)

        print(f"✅ Processed and visualized image index {idx}")
        if idx == 5:
            break  # Only process the first image

# Example usage
pickle_path = "foot_data.pkl"  # Update with your pickle file path
data = load_pickle_file(pickle_path)
extract_patches_and_visualize(data)
