import os
import json
import logging
import torch
import numpy as np
import pandas as pd
from collections import Counter, defaultdict
from datetime import datetime
from PIL import Image
import copy

from torch.utils.data import Subset
from torchvision import transforms

import _init_path
from config import cfg, update_config
from dataset.joint_patches import FinalSamplesDataset
from utils.utils import stratified_split_dataset, check_label_distribution_from_subset, split_dataset_by_patient_id_and_class, get_xray_augmentation

def setup_logger(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    # File handler
    file_handler = logging.FileHandler(os.path.join(output_dir, 'seed_analysis.log'))
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger

def analyze_seed_dataset(seed: int, cfg):
    """특정 시드의 데이터셋을 분석"""
    cfg.defrost()
    cfg.DATASET.SEED = seed
    cfg.freeze()

    # 1. transform 정의
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
    ])
    
    val_test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
    ])

    # 2. 전체 데이터셋 생성 (인덱스 추출용)
    dataset = FinalSamplesDataset(cfg, image_transform=val_test_transform)

    # 3. 리스트로 변환 및 split
    data_list = list(dataset.data.values())
    train_data, val_data, test_data = split_dataset_by_patient_id_and_class(
        data_list, val_size=0.2, test_size=0.2, random_state=seed
    )

    # 4. 인덱스 추출 함수
    def get_indices_from_entries(dataset, entries):
        entry_set = set((e['patient_id'], e['file_path']) for e in entries)
        indices = [
            idx for idx, v in dataset.data.items()
            if (v['patient_id'], v['file_path']) in entry_set
        ]
        return indices

    train_indices = get_indices_from_entries(dataset, train_data)
    val_indices = get_indices_from_entries(dataset, val_data)
    test_indices = get_indices_from_entries(dataset, test_data)

    # 5. subset별로 transform 다르게 적용
    # train: 증강 포함 (1:1 비율)
    train_dataset = FinalSamplesDataset(
        cfg, 
        image_transform=train_transform,
        augment_transform=get_xray_augmentation(),
        augment_ratio=1  # 1:1 비율로 증강
    )
    
    # val/test: 증강 없음
    val_dataset = FinalSamplesDataset(
        cfg, 
        image_transform=val_test_transform,
        augment_transform=None,
        augment_ratio=0
    )
    
    test_dataset = FinalSamplesDataset(
        cfg, 
        image_transform=val_test_transform,
        augment_transform=None,
        augment_ratio=0
    )

    # 6. Subset 생성
    train_subset = Subset(train_dataset, train_indices)
    val_subset = Subset(val_dataset, val_indices)
    test_subset = Subset(test_dataset, test_indices)

    # 7. 크기 계산
    total_size = len(dataset)
    train_size = len(train_subset)  # 원본 + 증강본 포함
    val_size = len(val_subset)
    test_size = len(test_subset)

    # 8. Patient ID 수집 (원본 기준)
    train_patient_ids = [train_dataset.data[i]['patient_id'] for i in train_indices]
    val_patient_ids = [val_dataset.data[i]['patient_id'] for i in val_indices]
    test_patient_ids = [test_dataset.data[i]['patient_id'] for i in test_indices]

    # 9. 클래스별 분포 수집 (원본 기준)
    train_labels = [train_dataset.data[i]['class_label'] for i in train_indices]
    val_labels = [val_dataset.data[i]['class_label'] for i in val_indices]
    test_labels = [test_dataset.data[i]['class_label'] for i in test_indices]

    # 10. 중복 확인
    train_val_overlap = set(train_patient_ids) & set(val_patient_ids)
    train_test_overlap = set(train_patient_ids) & set(test_patient_ids)
    val_test_overlap = set(val_patient_ids) & set(test_patient_ids)

    return {
        'seed': seed,
        'total_size': total_size,
        'train_size': train_size,  # 원본 + 증강본 포함
        'val_size': val_size,
        'test_size': test_size,
        'train_patient_ids': train_patient_ids,
        'val_patient_ids': val_patient_ids,
        'test_patient_ids': test_patient_ids,
        'train_labels': train_labels,
        'val_labels': val_labels,
        'test_labels': test_labels,
        'train_val_overlap': len(train_val_overlap),
        'train_test_overlap': len(train_test_overlap),
        'val_test_overlap': len(val_test_overlap),
        'train_val_overlap_patients': list(train_val_overlap),
        'train_test_overlap_patients': list(train_test_overlap),
        'val_test_overlap_patients': list(val_test_overlap)
    }

def comprehensive_seed_analysis(cfg, num_seeds=20):
    """0~19 시드까지 종합 분석"""
    logger = logging.getLogger()
    
    logger.info("🔍 [Seed 0-19 종합 분석 시작]")
    logger.info("=" * 80)

    all_results = []
    
    # 1. 각 시드별 기본 분석
    logger.info("📊 1단계: 각 시드별 기본 분석")
    logger.info("-" * 50)
    
    for seed in range(num_seeds):
        logger.info(f"🔍 Seed {seed} 분석 중...")
        result = analyze_seed_dataset(seed, cfg)
        all_results.append(result)
        
        logger.info(f"  📈 크기: Total={result['total_size']}, Train={result['train_size']}, Val={result['val_size']}, Test={result['test_size']}")
        logger.info(f"  🔗 중복: Train-Val={result['train_val_overlap']}, Train-Test={result['train_test_overlap']}, Val-Test={result['val_test_overlap']}")

    # 2. 데이터 크기 일관성 검사
    logger.info("\n📊 2단계: 데이터 크기 일관성 검사")
    logger.info("-" * 50)
    
    sizes_df = pd.DataFrame([{
        'seed': r['seed'],
        'total': r['total_size'],
        'train': r['train_size'],
        'val': r['val_size'],
        'test': r['test_size']
    } for r in all_results])
    
    logger.info(f"📏 크기 통계:")
    logger.info(f"  Total: mean={sizes_df['total'].mean():.1f}, std={sizes_df['total'].std():.1f}")
    logger.info(f"  Train: mean={sizes_df['train'].mean():.1f}, std={sizes_df['train'].std():.1f}")
    logger.info(f"  Val:   mean={sizes_df['val'].mean():.1f}, std={sizes_df['val'].std():.1f}")
    logger.info(f"  Test:  mean={sizes_df['test'].mean():.1f}, std={sizes_df['test'].std():.1f}")
    
    # 크기 이상치 확인
    for col in ['total', 'train', 'val', 'test']:
        mean_val = sizes_df[col].mean()
        std_val = sizes_df[col].std()
        outliers = sizes_df[abs(sizes_df[col] - mean_val) > 2 * std_val]
        if not outliers.empty:
            logger.warning(f"⚠️  {col} 크기 이상치: {outliers[['seed', col]].to_dict('records')}")

    # 3. 중복 데이터 검사
    logger.info("\n📊 3단계: 중복 데이터 검사")
    logger.info("-" * 50)
    
    overlap_df = pd.DataFrame([{
        'seed': r['seed'],
        'train_val_overlap': r['train_val_overlap'],
        'train_test_overlap': r['train_test_overlap'],
        'val_test_overlap': r['val_test_overlap']
    } for r in all_results])
    
    logger.info(f"🔗 중복 통계:")
    logger.info(f"  Train-Val 중복: mean={overlap_df['train_val_overlap'].mean():.1f}, max={overlap_df['train_val_overlap'].max()}")
    logger.info(f"  Train-Test 중복: mean={overlap_df['train_test_overlap'].mean():.1f}, max={overlap_df['train_test_overlap'].max()}")
    logger.info(f"  Val-Test 중복: mean={overlap_df['val_test_overlap'].mean():.1f}, max={overlap_df['val_test_overlap'].max()}")
    
    # 중복이 있는 시드들 확인
    problematic_seeds = overlap_df[
        (overlap_df['train_val_overlap'] > 0) | 
        (overlap_df['train_test_overlap'] > 0) | 
        (overlap_df['val_test_overlap'] > 0)
    ]
    
    if not problematic_seeds.empty:
        logger.warning(f"⚠️  중복이 있는 시드들: {problematic_seeds.to_dict('records')}")
    else:
        logger.info("✅ 모든 시드에서 중복 없음")

    # 4. 클래스 분포 검사
    logger.info("\n📊 4단계: 클래스 분포 검사")
    logger.info("-" * 50)
    
    for seed in range(num_seeds):
        result = all_results[seed]
        logger.info(f"🔍 Seed {seed} 클래스 분포:")
        
        train_dist = Counter(result['train_labels'])
        val_dist = Counter(result['val_labels'])
        test_dist = Counter(result['test_labels'])
        
        logger.info(f"  Train: {dict(train_dist)}")
        logger.info(f"  Val:   {dict(val_dist)}")
        logger.info(f"  Test:  {dict(test_dist)}")
        
        # 극단적인 분포 확인
        for split_name, dist in [('Train', train_dist), ('Val', val_dist), ('Test', test_dist)]:
            if len(dist) > 1:  # 이진 분류인 경우
                min_count = min(dist.values())
                max_count = max(dist.values())
                ratio = max_count / min_count if min_count > 0 else float('inf')
                if ratio > 3:  # 3:1 이상의 불균형
                    logger.warning(f"⚠️  Seed {seed} {split_name} 불균형: {ratio:.1f}:1")

    # 5. 시드 간 일관성 검사
    logger.info("\n📊 5단계: 시드 간 일관성 검사")
    logger.info("-" * 50)
    
    # Test 세트 크기 변동성
    test_sizes = [r['test_size'] for r in all_results]
    test_size_std = np.std(test_sizes)
    test_size_cv = test_size_std / np.mean(test_sizes)  # 변동계수
    
    logger.info(f"📏 Test 세트 크기 변동성:")
    logger.info(f"  표준편차: {test_size_std:.1f}")
    logger.info(f"  변동계수: {test_size_cv:.3f}")
    
    if test_size_cv > 0.1:  # 10% 이상 변동
        logger.warning(f"⚠️  Test 세트 크기가 불안정 (변동계수: {test_size_cv:.3f})")
    
    # 6. 특이한 시드 식별
    logger.info("\n📊 6단계: 특이한 시드 식별")
    logger.info("-" * 50)
    
    # Test 세트가 가장 작은 시드
    min_test_seed = min(all_results, key=lambda x: x['test_size'])
    logger.info(f"🔍 Test 세트가 가장 작은 시드: {min_test_seed['seed']} (크기: {min_test_seed['test_size']})")
    
    # Test 세트가 가장 큰 시드
    max_test_seed = max(all_results, key=lambda x: x['test_size'])
    logger.info(f"🔍 Test 세트가 가장 큰 시드: {max_test_seed['seed']} (크기: {max_test_seed['test_size']})")
    
    # 중복이 가장 많은 시드
    max_overlap_seed = max(all_results, key=lambda x: x['train_test_overlap'])
    if max_overlap_seed['train_test_overlap'] > 0:
        logger.warning(f"⚠️  Train-Test 중복이 가장 많은 시드: {max_overlap_seed['seed']} (중복: {max_overlap_seed['train_test_overlap']})")

    # 7. 결과 요약
    logger.info("\n📊 7단계: 종합 결과 요약")
    logger.info("=" * 80)
    
    # 문제가 있는 시드들 식별
    problematic_seeds = []
    
    for result in all_results:
        issues = []
        
        # Test 세트가 너무 작은 경우
        if result['test_size'] < 20:
            issues.append(f"Test세트작음({result['test_size']})")
        
        # 중복이 있는 경우
        if result['train_test_overlap'] > 0:
            issues.append(f"Train-Test중복({result['train_test_overlap']})")
        
        if result['train_val_overlap'] > 0:
            issues.append(f"Train-Val중복({result['train_val_overlap']})")
        
        if issues:
            problematic_seeds.append({
                'seed': result['seed'],
                'issues': issues,
                'test_size': result['test_size']
            })
    
    if problematic_seeds:
        logger.warning(f"⚠️  문제가 있는 시드들:")
        for ps in problematic_seeds:
            logger.warning(f"  Seed {ps['seed']}: {', '.join(ps['issues'])} (Test크기: {ps['test_size']})")
    else:
        logger.info("✅ 모든 시드가 정상")

    # 8. 권장사항
    logger.info("\n📋 8단계: 권장사항")
    logger.info("=" * 80)
    
    if problematic_seeds:
        logger.info("🔧 권장 조치:")
        logger.info("  1. 문제가 있는 시드들을 제외하고 실험 재실행")
        logger.info("  2. Test 세트 크기가 20개 미만인 시드들은 제외")
        logger.info("  3. 중복이 있는 시드들은 데이터 누출 가능성이 있으므로 제외")
        logger.info("  4. Patient ID 기반으로 중복을 완전히 제거하는 로직 추가 고려")
    else:
        logger.info("✅ 모든 시드가 정상이므로 현재 설정으로 실험 진행 가능")

    return all_results, problematic_seeds

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', default='config/large/tmp/origin_oa_normal.yaml', type=str)
    parser.add_argument('--num_seeds', default=20, type=int)
    args = parser.parse_args()

    update_config(cfg, args)
    
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    output_dir = os.path.join('output', f"seed_analysis_{timestamp}")
    logger = setup_logger(output_dir)
    
    logger.info(f"🔍 Seed 0-{args.num_seeds-1} 종합 분석 시작")
    logger.info(f"📁 설정 파일: {args.cfg}")
    logger.info(f"📁 출력 디렉토리: {output_dir}")
    
    all_results, problematic_seeds = comprehensive_seed_analysis(cfg, args.num_seeds)
    
    # 결과를 CSV로 저장
    results_df = pd.DataFrame([{
        'seed': r['seed'],
        'total_size': r['total_size'],
        'train_size': r['train_size'],
        'val_size': r['val_size'],
        'test_size': r['test_size'],
        'train_val_overlap': r['train_val_overlap'],
        'train_test_overlap': r['train_test_overlap'],
        'val_test_overlap': r['val_test_overlap']
    } for r in all_results])
    
    csv_path = os.path.join(output_dir, 'seed_analysis_results.csv')
    results_df.to_csv(csv_path, index=False)
    logger.info(f"📄 결과가 CSV로 저장되었습니다: {csv_path}")
    
    # 문제가 있는 시드들만 별도 저장
    if problematic_seeds:
        problematic_df = pd.DataFrame(problematic_seeds)
        problematic_csv_path = os.path.join(output_dir, 'problematic_seeds.csv')
        problematic_df.to_csv(problematic_csv_path, index=False)
        logger.info(f"📄 문제 시드들이 별도로 저장되었습니다: {problematic_csv_path}")

if __name__ == '__main__':
    main()