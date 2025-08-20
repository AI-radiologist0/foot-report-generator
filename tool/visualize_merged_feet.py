import sys
import os, argparse
import _init_path
import cv2
from dataset.joint_patches import FinalSamplesDataset
from config import cfg, update_config


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='config/large/tmp/proj/swin_t_resnet/origin_oa_normal_sampling20_proj_linear.yaml')
    parser.add_argument('--samples_per_class', type=int, default=10)
    parser.add_argument('--output_dir', type=str, default='merged_feet_samples', help='결과 이미지를 저장할 디렉토리')
    return parser.parse_args()

def save_merged_feet_samples():
    """is_same_image 값에 따라 같은 이미지와 서로 다른 이미지에서 온 발들을 각각 설정된 샘플 수만큼 저장"""
    
    args = arg_parser()
    output_dir = args.output_dir
    
    try:
        # 설정 파일 import
        update_config(cfg, args)
        
        # 데이터셋 로드
        dataset = FinalSamplesDataset(cfg)
        
        # is_same_image 값에 따라 샘플 수집 및 저장
        same_image_samples = []  # 같은 이미지에서 온 샘플들
        different_image_samples = []  # 서로 다른 이미지에서 온 샘플들
        samples_per_class = args.samples_per_class
        
        print("데이터셋에서 같은 이미지와 서로 다른 이미지에서 온 발들을 각각 수집 및 저장 중...")
        for idx in range(len(dataset)):
            try:
                merged_img, meta, is_same_image = dataset.get_merged_feet_image(idx)
                class_label = meta['class_label']
                patient_id = meta['patient_id']
                
                # 같은 이미지에서 온 발들인지 서로 다른 이미지에서 온 발들인지 구분하여 저장
                if is_same_image:
                    # 같은 이미지에서 온 발들
                    if len(same_image_samples) < samples_per_class:
                        sample_num = len(same_image_samples) + 1
                        save_path = os.path.join(output_dir, "same_image", f"same_image_sample_{sample_num:02d}.jpg")
                        
                        # 샘플 정보 저장
                        sample_info = {
                            'image': merged_img,
                            'meta': meta,
                            'idx': idx,
                            'patient_id': patient_id,
                            'class_label': class_label
                        }
                        same_image_samples.append(sample_info)
                        
                        # 이미지 저장
                        os.makedirs(os.path.dirname(save_path), exist_ok=True)
                        cv2.imwrite(save_path, merged_img)
                        print(f"같은 이미지에서 온 발들 - 샘플 {sample_num} 저장: {save_path} (환자: {patient_id}, 클래스: {class_label})")
                else:
                    # 서로 다른 이미지에서 온 발들
                    if len(different_image_samples) < samples_per_class:
                        sample_num = len(different_image_samples) + 1
                        save_path = os.path.join(output_dir, "different_image", f"different_image_sample_{sample_num:02d}.jpg")
                        
                        # 샘플 정보 저장
                        sample_info = {
                            'image': merged_img,
                            'meta': meta,
                            'idx': idx,
                            'patient_id': patient_id,
                            'class_label': class_label
                        }
                        different_image_samples.append(sample_info)
                        
                        # 이미지 저장
                        os.makedirs(os.path.dirname(save_path), exist_ok=True)
                        cv2.imwrite(save_path, merged_img)
                        print(f"서로 다른 이미지에서 온 발들 - 샘플 {sample_num} 저장: {save_path} (환자: {patient_id}, 클래스: {class_label})")
                
                # 두 그룹 모두 충분한 샘플을 모으면 종료
                if len(same_image_samples) >= samples_per_class and len(different_image_samples) >= samples_per_class:
                    print("두 그룹 모두 충분한 샘플을 수집했습니다. 종료합니다.")
                    break
                    
            except Exception as e:
                print(f"Error processing index {idx}: {e}")
                continue
        
        # 저장 완료 요약
        print(f"\n=== 저장 완료 요약 ===")
        print(f"같은 이미지에서 온 발들: {len(same_image_samples)}개 (목표: {samples_per_class}개)")
        print(f"서로 다른 이미지에서 온 발들: {len(different_image_samples)}개 (목표: {samples_per_class}개)")
        
        # 환자별 통계
        same_patients = set(sample['patient_id'] for sample in same_image_samples)
        different_patients = set(sample['patient_id'] for sample in different_image_samples)
        print(f"\n같은 이미지 그룹에 포함된 환자 수: {len(same_patients)}명")
        print(f"서로 다른 이미지 그룹에 포함된 환자 수: {len(different_patients)}명")
        
        # 클래스별 통계
        same_classes = {}
        different_classes = {}
        for sample in same_image_samples:
            class_label = sample['class_label']
            same_classes[class_label] = same_classes.get(class_label, 0) + 1
        
        for sample in different_image_samples:
            class_label = sample['class_label']
            different_classes[class_label] = different_classes.get(class_label, 0) + 1
        
        print(f"\n=== 클래스별 분포 ===")
        print("같은 이미지 그룹:")
        for class_label, count in same_classes.items():
            print(f"  클래스 {class_label}: {count}개")
        
        print("서로 다른 이미지 그룹:")
        for class_label, count in different_classes.items():
            print(f"  클래스 {class_label}: {count}개")
                
    except ImportError as e:
        print(f"설정 파일을 찾을 수 없습니다: {e}")
        print("config_path 변수를 실제 설정 파일 경로로 수정해주세요.")
    except Exception as e:
        print(f"오류 발생: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    save_merged_feet_samples()