import _init_path
import argparse
import os
from config import cfg, update_config
from dataset.joint_patches import FinalSamplesDataset
from tqdm import tqdm
import cv2


def main():
    parser = argparse.ArgumentParser(description="Save merged (cropped) feet images for all patients in FinalSamplesDataset.")
    parser.add_argument('--config', type=str, default="configs/final_sample_config.py", help='Path to config file')
    parser.add_argument('--output_dir', type=str, default="output/merged_feet", help='Directory to save merged images')
    parser.add_argument('--max_samples', type=int, default=None, help='Max number of samples to process')
    args = parser.parse_args()

    args.cfg = args.config
    update_config(cfg, args)
    dataset = FinalSamplesDataset(cfg)

    os.makedirs(args.output_dir, exist_ok=True)

    num_samples = len(dataset.indices)
    if args.max_samples is not None:
        num_samples = min(num_samples, args.max_samples)
    sizes = []
    for idx in tqdm(range(num_samples), desc="Saving merged feet images"):
        merged_img, meta = dataset.get_merged_feet_image(idx)
        patient_id = meta.get('patient_id', f'idx_{idx}')
        class_label = meta.get('class_label', 'unknown')
        filename = f"{idx:04d}_{patient_id}_{class_label}.png"
        save_path = os.path.join(args.output_dir, filename)
        cv2.imwrite(save_path, cv2.cvtColor(merged_img, cv2.COLOR_RGB2BGR))
        h, w = merged_img.shape[:2]
        sizes.append((w, h))
        # except Exception as e:
        #     print(f"[ERROR] idx={idx}: {e}")

    if sizes:
        import numpy as np
        widths, heights = zip(*sizes)
        widths = np.array(widths)
        heights = np.array(heights)
        print(f"총 {len(sizes)}개 샘플 저장")
        print(f"최소 해상도: {widths.min()}x{heights.min()}")
        print(f"최대 해상도: {widths.max()}x{heights.max()}")
        print(f"평균 해상도: {widths.mean():.1f}x{heights.mean():.1f}")
    else:
        print("저장된 이미지가 없습니다.")

    print(f"Done! Saved {num_samples} merged images to {args.output_dir}")

if __name__ == "__main__":
    main() 