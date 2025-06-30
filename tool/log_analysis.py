import os
import re
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.stats import norm
import torch
import torch.utils.data as torchUtils
import wandb
from torch.utils.tensorboard import SummaryWriter

import _init_path
from config import cfg, update_config
from utils.vis import plot_roc_curve
from core.patch_trainer import PatchTrainer
from dataset.joint_patches import FinalSamplesDataset
from utils.utils import stratified_split_dataset
import models


def extract_disease_set(paths) -> set:
    
    disease_set = None
    for path in paths:
        tmp = set(re.findall(r"(gout|oa|normal|ra|uncertain)", path.lower()))
        if disease_set is None:
            disease_set = tmp
        else:
            if disease_set != tmp:
                raise Exception(f"Error Occur!")
                break
        
    return disease_set


def flatten_binary_scores(y_score_list):
    return [float(s[0]) if isinstance(s, (list, np.ndarray)) and len(s) == 1 else float(s)
            for s in y_score_list]


def analyze_log_and_print_stats(log: str, output_dir: str = None):
    print("🔍 Parsing log file and extracting metrics...")
    with open(log, "r", encoding="utf-8") as f:
        lines = f.readlines()

    results = []
    wandb_paths = {}
    current_seed = None
    pending_wandb_path = None

    for i in range(len(lines) - 1):
        line = lines[i].strip()
        # ✅ seed 추출
        seed_match = re.search(r"\[Seed=(\d+)\]", line)
        if seed_match:
            current_seed = int(seed_match.group(1))
            if pending_wandb_path:
                wandb_paths[current_seed] = pending_wandb_path
                pending_wandb_path = None

        # ✅ wandb 로그 경로 추출
        if "wandb: Find logs at:" in line:
            path_match = re.search(r"wandb: Find logs at:\s*(.+)", line)
            if path_match:
                base_path = os.path.dirname(path_match.group(1).strip())
                if current_seed is not None:
                    wandb_paths[current_seed] = base_path
                else:
                    pending_wandb_path = base_path
        if "[Epoch 999]" in line and "Validation Accuracy" in line:
            acc_match = re.search(r"Validation Accuracy: ([\d.]+)%", line)
            f1_line = lines[i + 1].strip() if i + 1 < len(lines) else ""
            pr_match = re.search(r"Precision: ([\d.]+)", f1_line)
            re_match = re.search(r"Recall: ([\d.]+)", f1_line)
            f1_match = re.search(r"F1 Score: ([\d.]+)", f1_line)

            if acc_match and pr_match and re_match and f1_match:
                results.append({
                    "seed": current_seed,
                    "accuracy": float(acc_match.group(1)),
                    "precision": float(pr_match.group(1)) * 100,
                    "recall": float(re_match.group(1)) * 100,
                    "f1": float(f1_match.group(1)) * 100
                })


    df = pd.DataFrame(results)
    print(f"✅ Found {len(df)} seed results from log.")

    if df.empty:
        print("❌ 로그에서 seed별 결과를 찾지 못했습니다. 로그 파일 포맷 또는 경로를 확인하세요.")
        exit(1)
    
    # ⚠️ seed가 None인 경우 0 ~ N-1로 보정
    if df["seed"].isnull().any():
        print("⚠️ seed 값이 누락되어 순차적으로 보정합니다.")
        df["seed"] = list(range(len(df)))

    # ✅ 각 seed에 해당하는 best_model.pth 경로 추가
    df["pth"] = df["seed"].apply(lambda s: os.path.join(wandb_paths.get(s, ""), "files", "best_model.pth")) 

    # 통계 계산
    acc_mean = df["accuracy"].mean()
    acc_std = df["accuracy"].std()
    f1_mean = df["f1"].mean()
    f1_std = df["f1"].std()
    prec_mean = df["precision"].mean()
    prec_std = df["precision"].std()
    rec_mean = df["recall"].mean()
    rec_std = df["recall"].std()

    print("\n==== [Seed별 결과 통계] ====")
    print(f"Accuracy:  mean={acc_mean:.2f}%, std={acc_std:.2f}")
    print(f"Precision: mean={prec_mean:.2f}%, std={prec_std:.2f}")
    print(f"Recall:    mean={rec_mean:.2f}%, std={rec_std:.2f}")
    print(f"F1:        mean={f1_mean:.2f}%, std={f1_std:.2f}")
    print("==========================\n")

    # 히스토그램 저장 (output_dir이 주어졌을 때만)
    if output_dir is not None:
        counts, bins, _ = plt.hist(df["accuracy"], bins=10, color='lightsteelblue', edgecolor='black', alpha=0.7)
        bin_centers = 0.5 * (bins[1:] + bins[:-1])
        pdf = norm.pdf(bin_centers, acc_mean, df["accuracy"].std())
        pdf_scaled = pdf * (counts.sum() * np.diff(bins)[0])
        plt.plot(bin_centers, pdf_scaled, color='red', linewidth=2, label='Gaussian Fit')
        plt.axvline(acc_mean, color='blue', linestyle='--', linewidth=1.5, label=f"Mean = {acc_mean:.2f}%")
        plt.xlabel("Accuracy (%)")
        plt.ylabel("Frequency")
        plt.title("Accuracy Histogram")
        plt.gca().yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        plt.grid(axis='y', linestyle='--', alpha=0.6)
        plt.legend()
        hist_path = os.path.join(output_dir, "accuracy_histogram.png")
        plt.tight_layout()
        plt.savefig(hist_path)
        plt.close()
        print(f"📊 Accuracy histogram saved to: {hist_path}")

    return df, wandb_paths


def run_model_test_and_visualize(df, wandb_paths, cfg, final_output_dir, device, disease_name):
    # 대표 seed 결정
    acc_mean = df["accuracy"].mean()
    f1_mean = df["f1"].mean()
    prec_mean = df["precision"].mean()
    rec_mean = df["recall"].mean()
    df["distance"] = np.sqrt(
        (df["accuracy"] - acc_mean)**2 +
        (df["f1"] - f1_mean)**2 +
        (df["precision"] - prec_mean)**2 +
        (df["recall"] - rec_mean)**2
    )
    best_row = df.loc[df["distance"].idxmin()]
    best_seed = int(best_row["seed"])
    print(f"📌 Best seed selected: {best_seed}")

    # 히스토그램 시각화
    counts, bins, _ = plt.hist(df["accuracy"], bins=10, color='lightsteelblue', edgecolor='black', alpha=0.7)
    bin_centers = 0.5 * (bins[1:] + bins[:-1])
    pdf = norm.pdf(bin_centers, acc_mean, df["accuracy"].std())
    pdf_scaled = pdf * (counts.sum() * np.diff(bins)[0])
    plt.plot(bin_centers, pdf_scaled, color='red', linewidth=2, label='Gaussian Fit')
    plt.axvline(acc_mean, color='blue', linestyle='--', linewidth=1.5, label=f"Mean = {acc_mean:.2f}%")
    plt.xlabel("Accuracy (%)")
    plt.ylabel("Frequency")
    plt.title("Accuracy Histogram")
    plt.gca().yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.legend()
    hist_path = os.path.join(final_output_dir, "accuracy_histogram.png")
    plt.tight_layout()
    plt.savefig(hist_path)
    plt.close()
    print(f"📊 Accuracy histogram saved to: {hist_path}")

    # 모델 로딩 및 ROC Curve 평가
    model_path = os.path.join(wandb_paths[best_seed], "files", "best_model.pth")
    print(f"📦 Loading model from: {model_path}")
    cfg.defrost()
    cfg.DATASET.SEED = best_seed
    cfg.freeze()

    dataset = FinalSamplesDataset(cfg)
    _, _, test_dataset = stratified_split_dataset(dataset, seed=best_seed)
    test_loader = torchUtils.DataLoader(test_dataset, batch_size=cfg.TEST.BATCH_SIZE_PER_GPU)

    model = eval(f"models.{cfg.MODEL.NAME}.get_feature_extractor")(cfg, is_train=False).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    writer_dict = {
        'writer': SummaryWriter(log_dir=os.path.join(final_output_dir, f'tensorboard/seed{best_seed}')),
        'train_global_steps': 0,
        'valid_global_steps': 0
    }

    trainer = PatchTrainer(cfg, model=model, output_dir=final_output_dir, writer_dict=writer_dict)
    acc, loss, precision, recall, f1, y_true, y_score = trainer.validate(999, model, test_loader)

    print(f"📊 Test Accuracy:  {acc:.2f}%")
    print(f"📊 Test Precision: {precision:.4f}")
    print(f"📊 Test Recall:    {recall:.4f}")
    print(f"📊 Test F1 Score:  {f1:.4f}")

    # ROC Curve 저장
    y_score = flatten_binary_scores(y_score) if trainer.is_binary else y_score
    roc_path = plot_roc_curve(y_true, y_score, final_output_dir, best_seed, trainer.is_binary)
    print(f"📈 ROC curve saved to: {roc_path}")

    # W&B 업로드
    run = wandb.init(project="classification-binary-eval", name=f"{disease_name} Evaluation", tags=[disease_name], reinit=True)
    run.log({
        "accuracy_histogram": wandb.Image(hist_path),
        "roc_curve": wandb.Image(roc_path),
        "best_seed": best_seed,
        "accuracy_mean": acc_mean,
        "accuracy_std": df["accuracy"].std(),
        "f1_mean": f1_mean,
        "f1_std": df["f1"].std(),
        "precision_mean": df["precision"].mean(),
        "precision_std": df["precision"].std(),
        "recall_mean": df["recall"].mean(),
        "recall_std": df["recall"].std()
    })
    run.finish()
    print("🚀 Logged results to Weights & Biases.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", type=str, nargs='+', required=True, help="분석할 로그 파일 경로 리스트")
    parser.add_argument("--exp_names", type=str, nargs='+', required=True, help="실험 이름 리스트 (로그 파일 순서와 일치)")
    parser.add_argument("--cfg", type=str, default="config/large/tmp/swin-t/origin_oa_normal_sampling20.yaml")
    parser.add_argument("--output_dir", type=str, default="sampling_output/histogram/swint")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--run_test", action="store_true", help="Run model test and visualization (default: False)")
    args = parser.parse_args()
    update_config(cfg, args)
    
    # 각 로그 파일별로 질환명 추출 및 검증
    disease_sets = []
    for log_path in args.log:
        disease_sets.append(extract_disease_set([log_path]))
    if not all(ds == disease_sets[0] for ds in disease_sets):
        print(f"❌ 로그 파일별 질환명이 다릅니다: {disease_sets}")
        exit(1)
    log_set = disease_sets[0]

    if hasattr(cfg.DATASET, "TARGET_CLASSES"):
        cfg_set = cfg.DATASET.TARGET_CLASSES
    elif hasattr(cfg.DATASET, "CLASS_NAMES"):
        cfg_set = cfg.DATASET.CLASS_NAMES
    if isinstance(cfg_set, str):
        tokens = re.split(r"[\\/_.\-]", cfg_set.lower())
    elif isinstance(cfg_set, list):
        tokens = [t.lower() for t in cfg_set]
    else:
        tokens = []
    print(tokens)
    cfg_set = set(tokens)
    if log_set != cfg_set:
        print(f"❌ 질환 불일치: log({log_set}) ≠ cfg({cfg_set})")
        exit(1)
    disease_name = "_".join(sorted(log_set))
    final_output_dir = os.path.join(args.output_dir, disease_name)
    os.makedirs(final_output_dir, exist_ok=True)
    print(f"🦶 Disease identified: {disease_name}")
    print(f"📁 Output directory created: {final_output_dir}")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    all_accs = {}
    all_f1s = {}
    all_recalls = {}
    all_precisions = {}
    for log_path, exp_name in zip(args.log, args.exp_names):
        print(f"🔍 Analyzing {exp_name}...")
        df, _ = analyze_log_and_print_stats(log_path, final_output_dir)
        all_accs[exp_name] = df["accuracy"].values
        all_f1s[exp_name] = df["f1"].values
        all_recalls[exp_name] = df["recall"].values
        all_precisions[exp_name] = df["precision"].values

    # 모든 지표를 하나의 DataFrame으로 저장
    metrics_dict = {}
    for metric_name, metric_dict in zip(
        ["accuracy", "f1", "recall", "precision"],
        [all_accs, all_f1s, all_recalls, all_precisions]
    ):
        for exp_name, values in metric_dict.items():
            metrics_dict[f"{exp_name}_{metric_name}"] = pd.Series(values)
    df_metrics = pd.DataFrame(metrics_dict)
    metrics_csv_path = os.path.join(final_output_dir, "metrics_boxplot_data.csv")
    df_metrics.to_csv(metrics_csv_path, index=False)
    print(f"📄 모든 지표 데이터가 CSV로 저장되었습니다: {metrics_csv_path}")

    # Boxplot 그리기 및 저장 (accuracy만)
    plt.figure(figsize=(8, 6))
    plt.boxplot(all_accs.values(), labels=all_accs.keys())
    plt.ylabel("Accuracy (%)")
    plt.title(f"{disease_name}: Model Comparison")
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    boxplot_path = os.path.join(final_output_dir, "accuracy_boxplot.png")
    plt.savefig(boxplot_path)
    plt.close()
    print(f"📦 Boxplot saved to: {boxplot_path}")

    # 여러 지표를 한 plot에 boxplot으로 그리기
    exp_names = list(all_accs.keys())
    n_exp = len(exp_names)
    x = np.arange(n_exp)
    width = 0.18  # 박스 폭
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']  # Accuracy, F1, Recall, Precision
    plt.figure(figsize=(10, 6))
    # Accuracy
    bp1 = plt.boxplot([all_accs[k] for k in exp_names], positions=x-width*1.5, widths=width, patch_artist=True,
                boxprops=dict(facecolor=colors[0], alpha=0.5), medianprops=dict(color='black'), labels=['']*n_exp)
    # F1
    bp2 = plt.boxplot([all_f1s[k] for k in exp_names], positions=x-width*0.5, widths=width, patch_artist=True,
                boxprops=dict(facecolor=colors[1], alpha=0.5), medianprops=dict(color='black'), labels=['']*n_exp)
    # Recall
    bp3 = plt.boxplot([all_recalls[k] for k in exp_names], positions=x+width*0.5, widths=width, patch_artist=True,
                boxprops=dict(facecolor=colors[2], alpha=0.5), medianprops=dict(color='black'), labels=['']*n_exp)
    # Precision
    bp4 = plt.boxplot([all_precisions[k] for k in exp_names], positions=x+width*1.5, widths=width, patch_artist=True,
                boxprops=dict(facecolor=colors[3], alpha=0.5), medianprops=dict(color='black'), labels=exp_names)
    plt.xticks(x, exp_names)
    plt.ylabel("Score (%)")
    plt.title(f"{disease_name}: Model Comparison (Accuracy, F1, Recall, Precision)")
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    from matplotlib.patches import Patch
    legend_handles = [Patch(facecolor=colors[i], edgecolor='black', label=label) for i, label in enumerate(['Accuracy', 'F1', 'Recall', 'Precision'])]
    plt.legend(handles=legend_handles)
    plt.tight_layout()
    all_metrics_boxplot_path = os.path.join(final_output_dir, "all_metrics_boxplot.png")
    plt.savefig(all_metrics_boxplot_path)
    plt.close()
    print(f"📦 All metrics boxplot saved to: {all_metrics_boxplot_path}")

    # 모델 테스트 및 시각화는 flag로 제어
    if args.run_test:
        run_model_test_and_visualize(df, wandb_paths, cfg, final_output_dir, device, disease_name)

    print("🎯 Analysis completed successfully.")


if __name__ == "__main__":
    main()
