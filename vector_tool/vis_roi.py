import os
import cv2
import pickle
import matplotlib.pyplot as plt

# 패치 저장 디렉토리
output_dir_patches = "data/patches/"
os.makedirs(output_dir_patches, exist_ok=True)

# 원본 이미지 시각화 저장 디렉토리
output_dir_visualization = "data/visualization/"
os.makedirs(output_dir_visualization, exist_ok=True)

# 저장된 패치 데이터 불러오기
with open("data/pkl/output200x300.pkl", 'rb') as pkl_file:
    final_data = pickle.load(pkl_file)

# 패치 및 원본 이미지 저장
for image_id, data in final_data.items():
    image_path = data["file_path"]
    patient_id = data.get("patient_id", "unknown")
    patches = data["bbx"]

    # 개별 패치 저장 디렉토리 생성
    patch_dir = os.path.join(output_dir_patches, str(image_id))
    os.makedirs(patch_dir, exist_ok=True)

    # 원본 이미지 로드
    image = cv2.imread(image_path)
    if image is None:
        print(f"⚠️ Could not read image: {image_path}")
        continue
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 시각화
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    plt.title(f"Image ID: {image_id} | Patient ID: {patient_id}")
    plt.axis('off')

    # 저장
    vis_filename = os.path.join(output_dir_visualization, f"{image_id}_visualization.png")
    plt.savefig(vis_filename)
    plt.close()

    for idx, patch in enumerate(patches):
        patch_filename = os.path.join(patch_dir, f"patch_{idx+1}.png")
        cv2.imwrite(patch_filename, cv2.cvtColor(patch, cv2.COLOR_RGB2BGR))

print(f"패치 저장 완료: {output_dir_patches}")
print(f"시각화 저장 완료: {output_dir_visualization}")
