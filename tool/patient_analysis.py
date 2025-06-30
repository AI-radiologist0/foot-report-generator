import os
import json
import logging
import pandas as pd
from collections import Counter, defaultdict
from datetime import datetime

def setup_logger(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    # File handler
    file_handler = logging.FileHandler(os.path.join(output_dir, 'patient_analysis.log'))
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger

def analyze_patient_distribution(json_path):
    """JSON 파일에서 환자별 분포 분석"""
    logger = logging.getLogger()
    
    logger.info(f"🔍 JSON 파일 분석 시작: {json_path}")
    logger.info("=" * 80)
    
    # JSON 파일 로드
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    logger.info(f"📊 총 데이터 개수: {len(data)}")
    
    # 환자별 데이터 수집
    patient_data = defaultdict(list)
    patient_classes = defaultdict(list)
    
    for idx, item in enumerate(data):
        patient_id = item.get('patient_id', 'unknown')
        class_label = item.get('class', 'unknown').lower()
        
        patient_data[patient_id].append({
            'index': idx,
            'class': class_label,
            'file_path': item.get('merged_image_path', ''),
            'left_right_files': item.get('file_paths', []),
            'diagnosis': item.get('diagnosis', '')
        })
        patient_classes[patient_id].append(class_label)
    
    # 환자별 통계
    patient_stats = {}
    for patient_id, items in patient_data.items():
        class_counts = Counter([item['class'] for item in items])
        patient_stats[patient_id] = {
            'total_images': len(items),
            'classes': dict(class_counts),
            'unique_classes': len(set(class_counts.keys())),
            'items': items
        }
    
    # 1. 기본 통계
    logger.info("📊 1단계: 기본 통계")
    logger.info("-" * 50)
    
    total_patients = len(patient_data)
    total_images = sum(stats['total_images'] for stats in patient_stats.values())
    
    logger.info(f"👥 총 환자 수: {total_patients}")
    logger.info(f"🖼️  총 이미지 수: {total_images}")
    logger.info(f"📈 환자당 평균 이미지 수: {total_images / total_patients:.2f}")
    
    # 2. 환자별 이미지 수 분포
    logger.info("\n📊 2단계: 환자별 이미지 수 분포")
    logger.info("-" * 50)
    
    image_counts = [stats['total_images'] for stats in patient_stats.values()]
    image_count_dist = Counter(image_counts)
    
    logger.info("환자별 이미지 수 분포:")
    for count, num_patients in sorted(image_count_dist.items()):
        logger.info(f"  {count}개 이미지: {num_patients}명 환자")
    
    # 3. 여러 이미지를 가진 환자들
    logger.info("\n📊 3단계: 여러 이미지를 가진 환자들")
    logger.info("-" * 50)
    
    multi_image_patients = {pid: stats for pid, stats in patient_stats.items() if stats['total_images'] > 1}
    
    logger.info(f"🔍 여러 이미지를 가진 환자 수: {len(multi_image_patients)}")
    
    if multi_image_patients:
        logger.info("📋 여러 이미지를 가진 환자 상세:")
        for patient_id, stats in sorted(multi_image_patients.items(), key=lambda x: x[1]['total_images'], reverse=True):
            logger.info(f"  환자 {patient_id}: {stats['total_images']}개 이미지")
            logger.info(f"    클래스 분포: {stats['classes']}")
            
            # 각 이미지 상세 정보
            for item in stats['items']:
                logger.info(f"    - 인덱스 {item['index']}: {item['class']} ({item['file_path']})")
            logger.info("")
    
    # 4. 같은 환자의 다른 클래스
    logger.info("\n📊 4단계: 같은 환자의 다른 클래스")
    logger.info("-" * 50)
    
    mixed_class_patients = {pid: stats for pid, stats in patient_stats.items() if stats['unique_classes'] > 1}
    
    logger.info(f"🔍 여러 클래스를 가진 환자 수: {len(mixed_class_patients)}")
    
    if mixed_class_patients:
        logger.info("📋 여러 클래스를 가진 환자 상세:")
        for patient_id, stats in sorted(mixed_class_patients.items(), key=lambda x: x[1]['unique_classes'], reverse=True):
            logger.info(f"  환자 {patient_id}: {stats['unique_classes']}개 클래스")
            logger.info(f"    클래스 분포: {stats['classes']}")
            logger.info("")
    
    # 5. 클래스별 환자 분포
    logger.info("\n📊 5단계: 클래스별 환자 분포")
    logger.info("-" * 50)
    
    class_patient_dist = defaultdict(set)
    for patient_id, classes in patient_classes.items():
        for class_name in classes:
            class_patient_dist[class_name].add(patient_id)
    
    logger.info("클래스별 환자 수:")
    for class_name, patients in sorted(class_patient_dist.items()):
        logger.info(f"  {class_name}: {len(patients)}명 환자")
    
    # 6. 중복 환자 ID 확인
    logger.info("\n📊 6단계: 중복 환자 ID 확인")
    logger.info("-" * 50)
    
    all_patient_ids = [item.get('patient_id', 'unknown') for item in data]
    duplicate_patient_ids = [pid for pid, count in Counter(all_patient_ids).items() if count > 1]
    
    if duplicate_patient_ids:
        logger.info(f"🔍 중복된 환자 ID 수: {len(duplicate_patient_ids)}")
        logger.info("📋 중복된 환자 ID들:")
        for pid in sorted(duplicate_patient_ids):
            count = all_patient_ids.count(pid)
            logger.info(f"  {pid}: {count}번 등장")
    else:
        logger.info("✅ 중복된 환자 ID 없음")
    
    # 7. 결과 요약
    logger.info("\n📊 7단계: 결과 요약")
    logger.info("=" * 80)
    
    logger.info("🔍 주요 발견사항:")
    
    # 가장 많은 이미지를 가진 환자
    if patient_stats:
        max_images_patient = max(patient_stats.items(), key=lambda x: x[1]['total_images'])
        logger.info(f"  📈 가장 많은 이미지: 환자 {max_images_patient[0]} ({max_images_patient[1]['total_images']}개)")
    
    # 가장 많은 클래스를 가진 환자
    if patient_stats:
        max_classes_patient = max(patient_stats.items(), key=lambda x: x[1]['unique_classes'])
        logger.info(f"  🎯 가장 많은 클래스: 환자 {max_classes_patient[0]} ({max_classes_patient[1]['unique_classes']}개)")
    
    # 데이터 누출 위험도
    multi_image_ratio = len(multi_image_patients) / total_patients
    logger.info(f"  ⚠️  데이터 누출 위험도: {multi_image_ratio:.1%} ({len(multi_image_patients)}/{total_patients} 환자)")
    
    if multi_image_ratio > 0.5:
        logger.warning("  🚨 높은 데이터 누출 위험! Patient ID 기반 분할 강력 권장")
    elif multi_image_ratio > 0.2:
        logger.warning("  ⚠️  중간 데이터 누출 위험! Patient ID 기반 분할 권장")
    else:
        logger.info("  ✅ 낮은 데이터 누출 위험")
    
    return patient_stats, multi_image_patients, mixed_class_patients

def save_analysis_results(patient_stats, output_dir):
    """분석 결과를 CSV로 저장"""
    logger = logging.getLogger()
    
    # 환자별 통계를 DataFrame으로 변환
    results = []
    for patient_id, stats in patient_stats.items():
        results.append({
            'patient_id': patient_id,
            'total_images': stats['total_images'],
            'unique_classes': stats['unique_classes'],
            'classes': str(stats['classes']),
            'has_multiple_images': stats['total_images'] > 1,
            'has_multiple_classes': stats['unique_classes'] > 1
        })
    
    df = pd.DataFrame(results)
    
    # CSV 저장
    csv_path = os.path.join(output_dir, 'patient_analysis_results.csv')
    df.to_csv(csv_path, index=False)
    logger.info(f"📄 분석 결과가 CSV로 저장되었습니다: {csv_path}")
    
    # 여러 이미지를 가진 환자들만 별도 저장
    multi_image_df = df[df['has_multiple_images'] == True]
    if not multi_image_df.empty:
        multi_image_csv_path = os.path.join(output_dir, 'multi_image_patients.csv')
        multi_image_df.to_csv(multi_image_csv_path, index=False)
        logger.info(f"📄 여러 이미지를 가진 환자들이 별도로 저장되었습니다: {multi_image_csv_path}")
    
    # 여러 클래스를 가진 환자들만 별도 저장
    multi_class_df = df[df['has_multiple_classes'] == True]
    if not multi_class_df.empty:
        multi_class_csv_path = os.path.join(output_dir, 'multi_class_patients.csv')
        multi_class_df.to_csv(multi_class_csv_path, index=False)
        logger.info(f"📄 여러 클래스를 가진 환자들이 별도로 저장되었습니다: {multi_class_csv_path}")

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--json', required=True, type=str, help='분석할 JSON 파일 경로')
    parser.add_argument('--output_dir', default=None, type=str, help='출력 디렉토리')
    args = parser.parse_args()
    
    # 출력 디렉토리 설정
    if args.output_dir is None:
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        json_name = os.path.splitext(os.path.basename(args.json))[0]
        output_dir = os.path.join('output', f"patient_analysis_{json_name}_{timestamp}")
    else:
        output_dir = args.output_dir
    
    logger = setup_logger(output_dir)
    
    logger.info(f"🔍 환자 분석 시작")
    logger.info(f"📁 JSON 파일: {args.json}")
    logger.info(f"📁 출력 디렉토리: {output_dir}")
    
    # 분석 실행
    patient_stats, multi_image_patients, mixed_class_patients = analyze_patient_distribution(args.json)
    
    # 결과 저장
    save_analysis_results(patient_stats, output_dir)
    
    logger.info("✅ 환자 분석 완료!")

if __name__ == '__main__':
    main()