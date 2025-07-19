import os
import json
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import yaml

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def check_image_channels(image_path):
    try:
        img = Image.open(image_path)
        original_mode = img.mode
        original_size = img.size
        img_rgb = img.convert("RGB")
        rgb_array = np.array(img_rgb)
        r_channel = rgb_array[:, :, 0]
        g_channel = rgb_array[:, :, 1]
        b_channel = rgb_array[:, :, 2]
        r_g_diff = np.abs(r_channel - g_channel)
        r_b_diff = np.abs(r_channel - b_channel)
        g_b_diff = np.abs(g_channel - b_channel)

        # 각 채널이 전체적으로 단일값인지
        is_r_uniform = np.all(r_channel == r_channel.flat[0])
        is_g_uniform = np.all(g_channel == g_channel.flat[0])
        is_b_uniform = np.all(b_channel == b_channel.flat[0])
        r_unique = np.unique(r_channel)
        g_unique = np.unique(g_channel)
        b_unique = np.unique(b_channel)
        is_all_channels_uniform = is_r_uniform and is_g_uniform and is_b_uniform

        result = {
            'file_path': image_path,
            'original_mode': original_mode,
            'original_size': original_size,
            'rgb_shape': rgb_array.shape,
            'r_g_max_diff': np.max(r_g_diff),
            'r_b_max_diff': np.max(r_b_diff),
            'g_b_max_diff': np.max(g_b_diff),
            'r_g_mean_diff': np.mean(r_g_diff),
            'r_b_mean_diff': np.mean(r_b_diff),
            'g_b_mean_diff': np.mean(g_b_diff),
            'is_grayscale': np.max(r_g_diff) == 0 and np.max(r_b_diff) == 0 and np.max(g_b_diff) == 0,
            'r_mean': np.mean(r_channel),
            'g_mean': np.mean(g_channel),
            'b_mean': np.mean(b_channel),
            'is_r_uniform': bool(is_r_uniform),
            'is_g_uniform': bool(is_g_uniform),
            'is_b_uniform': bool(is_b_uniform),
            'r_unique_count': int(len(r_unique)),
            'g_unique_count': int(len(g_unique)),
            'b_unique_count': int(len(b_unique)),
            'is_all_channels_uniform': bool(is_all_channels_uniform)
        }
        return result
    except Exception as e:
        return {
            'file_path': image_path,
            'error': str(e)
        }

def resolve_image_path(image_path, json_path, project_root=None):
    # 절대경로면 그대로
    if os.path.isabs(image_path):
        return image_path
    # ./로 시작하면 프로젝트 루트 기준
    if image_path.startswith('./'):
        if project_root is None:
            project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        return os.path.normpath(os.path.join(project_root, image_path[2:]))
    # 그 외는 json 파일 기준 상대경로
    json_dir = os.path.dirname(json_path)
    return os.path.normpath(os.path.join(json_dir, image_path))

def analyze_dataset_with_config(config_path, max_samples=100):
    print(f"설정 파일 로드: {config_path}")
    config = load_config(config_path)
    json_path = config.get('DATASET', {}).get('JSON', '')
    if not json_path:
        print("설정 파일에서 JSON 경로를 찾을 수 없습니다.")
        return None
    if not os.path.isabs(json_path):
        config_dir = os.path.dirname(config_path)
        json_path = os.path.join(config_dir, json_path)
    print(f"JSON 파일 경로: {json_path}")
    if not os.path.exists(json_path):
        print(f"JSON 파일을 찾을 수 없습니다: {json_path}")
        return None
    return analyze_dataset_channels(json_path, max_samples)

def analyze_dataset_channels(json_path, max_samples=100):
    print(f"데이터셋 분석 시작: {json_path}")
    with open(json_path, 'r') as f:
        data = json.load(f)
    print(f"총 {len(data)}개 샘플 중 {min(max_samples, len(data))}개 분석")
    results = []
    grayscale_count = 0
    non_grayscale_count = 0
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    for i, item in tqdm(enumerate(data[:max_samples]), total=min(max_samples, len(data))):
        if 'merged_image_path' in item:
            image_path = item['merged_image_path']
        elif 'file_path' in item:
            image_path = item['file_path']
        else:
            continue
        image_path = resolve_image_path(image_path, json_path, project_root)
        result = check_image_channels(image_path)
        results.append(result)
        if 'error' not in result:
            if result['is_grayscale']:
                grayscale_count += 1
            else:
                non_grayscale_count += 1
    print(f"\n=== 분석 결과 ===")
    print(f"총 분석된 이미지: {len(results)}")
    print(f"Grayscale 이미지: {grayscale_count}")
    print(f"Non-grayscale 이미지: {non_grayscale_count}")
    print(f"Grayscale 비율: {grayscale_count/len(results)*100:.2f}%")
    non_grayscale_results = [r for r in results if 'error' not in r and not r['is_grayscale']]
    if non_grayscale_results:
        print(f"\n=== Non-grayscale 이미지 상세 정보 ===")
        for i, result in enumerate(non_grayscale_results[:5]):
            print(f"\n{i+1}. {os.path.basename(result['file_path'])}")
            print(f"   R-G 최대 차이: {result['r_g_max_diff']}")
            print(f"   R-B 최대 차이: {result['r_b_max_diff']}")
            print(f"   G-B 최대 차이: {result['g_b_max_diff']}")
            print(f"   R 평균: {result['r_mean']:.2f}")
            print(f"   G 평균: {result['g_mean']:.2f}")
            print(f"   B 평균: {result['b_mean']:.2f}")
            print(f"   R 고유값 개수: {result['r_unique_count']}, G 고유값 개수: {result['g_unique_count']}, B 고유값 개수: {result['b_unique_count']}")
            print(f"   R 단일값: {result['is_r_uniform']}, G 단일값: {result['is_g_uniform']}, B 단일값: {result['is_b_uniform']}, 전체채널단일: {result['is_all_channels_uniform']}")
    error_results = [r for r in results if 'error' in r]
    if error_results:
        print(f"\n=== 에러가 발생한 이미지들 ===")
        for result in error_results:
            print(f"  {os.path.basename(result['file_path'])}: {result['error']}")
    return results

def visualize_sample_images(json_path, num_samples=5):
    with open(json_path, 'r') as f:
        data = json.load(f)
    fig, axes = plt.subplots(num_samples, 4, figsize=(20, 5*num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    for i in range(min(num_samples, len(data))):
        item = data[i]
        if 'merged_image_path' in item:
            image_path = item['merged_image_path']
        elif 'file_path' in item:
            image_path = item['file_path']
        else:
            continue
        if not os.path.isabs(image_path):
            json_dir = os.path.dirname(json_path)
            image_path = os.path.join(json_dir, image_path)
        try:
            img = Image.open(image_path)
            original_img = img.convert("RGB")
            r, g, b = original_img.split()
            axes[i, 0].imshow(original_img)
            axes[i, 0].set_title(f'Original RGB\n{os.path.basename(image_path)}')
            axes[i, 0].axis('off')
            axes[i, 1].imshow(r, cmap='Reds')
            axes[i, 1].set_title('Red Channel')
            axes[i, 1].axis('off')
            axes[i, 2].imshow(g, cmap='Greens')
            axes[i, 2].set_title('Green Channel')
            axes[i, 2].axis('off')
            axes[i, 3].imshow(b, cmap='Blues')
            axes[i, 3].set_title('Blue Channel')
            axes[i, 3].axis('off')
        except Exception as e:
            axes[i, 0].text(0.5, 0.5, f'Error: {str(e)}', ha='center', va='center')
            axes[i, 0].set_title(f'Error\n{os.path.basename(image_path)}')
    plt.tight_layout()
    plt.savefig('channel_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='X-ray 이미지 채널 분석')
    parser.add_argument('--config', type=str, help='설정 파일 경로 (YAML)')
    parser.add_argument('--json', type=str, help='JSON 파일 경로')
    parser.add_argument('--max_samples', type=int, default=50, help='분석할 최대 샘플 수')
    parser.add_argument('--visualize', action='store_true', help='샘플 이미지 시각화')
    args = parser.parse_args()
    if args.config:
        results = analyze_dataset_with_config(args.config, args.max_samples)
        if args.visualize:
            config = load_config(args.config)
            json_path = config.get('DATASET', {}).get('JSON', '')
            if not os.path.isabs(json_path):
                config_dir = os.path.dirname(args.config)
                json_path = os.path.join(config_dir, json_path)
            visualize_sample_images(json_path, num_samples=5)
    elif args.json:
        results = analyze_dataset_channels(args.json, args.max_samples)
        if args.visualize:
            visualize_sample_images(args.json, num_samples=5)
    else:
        print("사용법:")
        print("  python tool/check_image_channels.py --config config/test.yaml --max_samples 20")
        print("  python tool/check_image_channels.py --json data/json/foot_ra_merge.json --max_samples 30 --visualize") 