import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
from pycocomedical import COCOMedical

class COCOMedicalDataset(Dataset):
    def __init__(self, json_path, bbox_size=120, transform=None):
        """
        Args:
            json_path (str): Path to the merged JSON file.
            bbox_size (int): Size of the patches to be extracted around keypoints.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.coco = COCOMedical(json_path)  # Load dataset using COCOMedical
        self.bbox_size = bbox_size
        self.transform = transform
        self.image_ids = list(self.coco.images.keys())  # List of image IDs
        self.patient_ids = list(self.coco.diseases.keys()) # list of disease info
        
        self.patches_db = []

    def __len__(self):
        return len(self.patches_db)
    
    def _get_db(self):
        if self.is_train:
            self.patches_db.extend(self._generate_patches()) # Generating patches for each patients.
    
    def _generate_patches(self):
        # load image_path 
        
        ret_list = []
        ret_bucket = {
            'image_path': None,
            'image_id': None,
            'left_patches': None,
            'right_patches': None,
            'patches': None,
            'class_label': None,
        }
        
        for id in self.patient_ids:
            image_info = self.coco.images.get(id)
            image_path = image_info.file_path
            
            image = cv2.imread(image_path, cv2.COLOR_RGB2BGR)

            disease_info = self.coco.diseases.get(id)
            
            class_label = disease_info['class_label']
            
            left_keypoint = disease_info['keypoints']
            
            
            
            
            
    def __getitem__(self, idx):
        pass

    def extract_patches(self, image, keypoints):
        """Extract patches around each keypoint."""
        
        height, width, _ = image.shape
        patches = [np.zeros((self.bbox_size, self.bbox_size, 3), dtype=np.uint8) for _ in range(17)]
        
        if len(keypoints) == 0:
            return patches
        
        for i in range(17):  # Joint 개수 17개 고정
            x, y, score = int(keypoints[i * 3]), int(keypoints[i * 3 + 1]), keypoints[i * 3 + 2]
        
            if score > 0.0:  # Valid keypoint
                x_min = max(x - self.bbox_size // 2, 0)
                y_min = max(y - self.bbox_size // 2, 0)
                x_max = min(x + self.bbox_size // 2, width)
                y_max = min(y + self.bbox_size // 2, height)

                cropped_patch = image[y_min:y_max, x_min:x_max]
                if (x_max < x_min) or (y_max < y_min):
                    return []
                try:
                    # Check if the cropped bbox is valid (non-empty)
                    if cropped_patch is not None and cropped_patch.size > 0:
                        resized_bbox = cv2.resize(cropped_patch, (self.bbox_size, self.bbox_size))
                    else:
                        print(f"⚠️ Warning: Empty cropped_bbox at ({x_min}, {y_min}) → Returning empty list.")
                        return []  # Return empty list if invalid region
                except cv2.error as e:
                    print(f"⚠️ OpenCV Resize Error: {e} → Returning empty list.")
                    return []  # Return empty list on error
                patches[i] = resized_bbox


        return np.array(patches, dtype=np.float32) / 255.0  # Normalize to [0,1]