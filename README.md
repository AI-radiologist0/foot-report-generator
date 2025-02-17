# Pickle File Structure for Foot Detection Data

## Overview

This pickle file stores processed foot detection data, including keypoints, extracted patches, and bounding box coordinates for detected feet. Each entry in the pickle file corresponds to a processed image, with metadata and extracted features.

## File Format

The pickle file is a Python dictionary where:

- Each **image index (integer)** is a key.
- The value is a nested dictionary containing metadata and extracted data.

## Structure

```
{
    0: {
        "meta": {
            "file_path": "path/to/image.jpg",
            "patient_id": "CAUHRA20004",
            "diagnosis": "hallux valgus, Rt ...",
            "class": "Uncertain"
        },
        "data": {
            "left_bbox": [x1, y1, x2, y2],  # Bounding box coordinates for the left foot
            "right_bbox": [x1, y1, x2, y2],  # Bounding box coordinates for the right foot
            "left_keypoints": [[x1, y1], [x2, y2], ..., [x17, y17]],  # List of left foot keypoints
            "right_keypoints": [[x1, y1], [x2, y2], ..., [x17, y17]],  # List of right foot keypoints
            "left_patches": {  # Dictionary of extracted patches for left foot
                1: [x1, y1, x2, y2],  # ROI coordinates for patch around keypoint 1
                2: [x1, y1, x2, y2],
                ...
                17: [x1, y1, x2, y2]
            },
            "right_patches": {  # Dictionary of extracted patches for right foot
                1: [x1, y1, x2, y2],  # ROI coordinates for patch around keypoint 1
                2: [x1, y1, x2, y2],
                ...
                17: [x1, y1, x2, y2]
            }
        }
    },
    1: { ... }  # Next processed image
}
```

## Description of Fields

- **meta**: Contains metadata about the image and patient.

  - `file_path`: Path to the original image.
  - `patient_id`: Unique patient identifier.
  - `diagnosis`: Doctor's diagnosis.
  - `class`: Diagnosis classification.

- **data**: Contains processed outputs.

  - `left_bbox`: Bounding box coordinates `[x1, y1, x2, y2]` for the left foot.
  - `right_bbox`: Bounding box coordinates `[x1, y1, x2, y2]` for the right foot.
  - `left_keypoints`: List of 17 (x, y) keypoints detected in the left foot.
  - `right_keypoints`: List of 17 (x, y) keypoints detected in the right foot.
  - `left_patches`: Dictionary mapping keypoint indices (1-17) to ROI coordinates `[x1, y1, x2, y2]` for the left foot.
  - `right_patches`: Dictionary mapping keypoint indices (1-17) to ROI coordinates `[x1, y1, x2, y2]` for the right foot.

## How to Load the Pickle File

To load the pickle file in Python:

```python
import pickle
import cv2

with open("foot_data.pkl", "rb") as f:
    data = pickle.load(f)

# Example: Accessing the first image entry
first_idx = list(data.keys())[0]  # Get the first index
image_data = data[first_idx]
print("Patient ID:", image_data["meta"]["patient_id"])
print("Left foot keypoints:", image_data["data"].get("left_keypoints", "Not detected"))

# Load the original image
image = cv2.imread(image_data["meta"]["file_path"])

# Crop the left foot from the original image using left_bbox
if "left_bbox" in image_data["data"]:
    x1, y1, x2, y2 = image_data["data"]["left_bbox"]
    left_cropped = image[y1:y2, x1:x2]
    cv2.imshow("Left Foot", left_cropped)
    cv2.waitKey(0)

# Extract patches from the left foot using left_patches
if "left_patches" in image_data["data"]:
    for keypoint_id, (px1, py1, px2, py2) in image_data["data"]["left_patches"].items():
        patch = left_cropped[py1:py2, px1:px2]
        cv2.imshow(f"Patch {keypoint_id}", patch)
        cv2.waitKey(0)

cv2.destroyAllWindows()
```

## Notes

- If **left or right foot is not detected**, the corresponding field will not be present in `data`.
- Instead of storing full patch images, the dataset now stores **ROI coordinates** `[x1, y1, x2, y2]` for patches.
- The bounding box coordinates **(********`left_bbox`********, ********`right_bbox`********)** store the detected foot region before resizing.
- To extract patches from the original image, use these coordinates with `cv2.imread()`.

## Conclusion

This structured format enables easy retrieval and analysis of foot detection data, allowing seamless integration into machine learning workflows or medical diagnosis tools.

