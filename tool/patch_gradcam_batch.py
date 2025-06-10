import os
import torch
import torch.nn as nn
import argparse
import matplotlib.pyplot as plt

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import BinaryClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

import _init_path
from config import cfg, update_config
from dataset.joint_patches import FinalSamplesDataset
from utils.utils import stratified_split_dataset
from models.feature_extractor import get_feature_extractor
from log_analysis import analyze_log_and_print_stats



def reshape_transform_swin_by_layer(tensor: torch.Tensor, layer_idx: int) -> torch.Tensor:
    if tensor.dim() != 4:
        raise ValueError(f"Expected 4D tensor, got {tensor.shape}")
    B, H, W, C = tensor.shape
    size_map = {
        0: (56, 56),
        1: (28, 28),
        2: (14, 14),
        3: (7, 7)
    }
    h, w = size_map.get(layer_idx, (28, 28))  # 기본 28x28
    return tensor.permute(0, 3, 1, 2).contiguous().view(B, C, h, w)


class ImagePatchWrapper(nn.Module):
    def __init__(self, model: nn.Module, patches: torch.Tensor):
        super().__init__()
        self.model = model
        self.patches = patches

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x, self.patches)


def get_best_model_path(log_file, output_dir):
    df, wandb_paths = analyze_log_and_print_stats(log_file, output_dir)
    best_row = df.loc[df["distance"].idxmin()] if "distance" in df.columns else df.iloc[df["accuracy"].idxmax()]
    best_seed = int(best_row["seed"])
    for ext in ["best_model.pth.tar", "best_model.pth"]:
        candidate = os.path.join(wandb_paths[best_seed], "files", ext)
        if os.path.exists(candidate):
            return candidate, best_seed
    raise FileNotFoundError("No best_model.pth(.tar) found for best_seed.")


def run_gradcam_single_block(
    image, patches, model, device,
    target_layer: nn.Module,
    reshape_transform_fn,
    output_dir: str,
    target_class: int,
    file_prefix: str = "",
    label: int = None,
    pred: int = None, 
    patient_id: str = None,
    class_label_mapping: dict = None,
    
):
    B = image.shape[0]

    cam = GradCAM(
        model=ImagePatchWrapper(model, patches),
        target_layers=[target_layer],
        reshape_transform=reshape_transform_fn
    )
    targets = [BinaryClassifierOutputTarget(category=target_class)] * B
    grayscale_cam = cam(input_tensor=image, targets=targets)

    os.makedirs(output_dir, exist_ok=True)

    for b in range(B):
        img = image[b].detach().cpu().numpy().transpose(1, 2, 0)
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
        cam_img = show_cam_on_image(img, grayscale_cam[b], use_rgb=True)
        
        # title 구성 
        gt_name = class_label_mapping.get(target_class, str(target_class))
        pred_name = class_label_mapping.get(pred, str(pred))
        title = f"GT: {gt_name} | Pred: {pred_name} | Patient ID: {patient_id}"
        plt.title(title, fontsize=10)  # title 추가

        # ✅ 새로운 figure를 명시적으로 생성 후 저장
        plt.figure(figsize=(4, 4))
        plt.axis('off')
        plt.imshow(cam_img)
        save_path = os.path.join(output_dir, f'{file_prefix}_b{b}_{patient_id}.png')
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
        plt.close()



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_file', type=str, default='sampling_output/log_sampling_gout_normal_sw_res.log')
    parser.add_argument('--cfg', type=str, default='config/large/tmp/swin_t_resnet/origin_gout_normal_sampling20.yaml')
    parser.add_argument('--output_dir', type=str, default='gradcam_layer1_block1_match_only')
    parser.add_argument('--num_samples', type=int, default=16)
    args = parser.parse_args()

    update_config(cfg, args)
    os.makedirs(args.output_dir, exist_ok=True)

    binary_name = '_'.join(cfg.DATASET.TARGET_CLASSES)
    best_model_path, best_seed = get_best_model_path(args.log_file, args.output_dir)
    print(f"Best model path: {best_model_path} (seed={best_seed})")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = FinalSamplesDataset(cfg)
    _, _, test_dataset = stratified_split_dataset(dataset, seed=best_seed)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.num_samples, shuffle=False)

    model = get_feature_extractor(cfg, is_train=False)
    model.load_state_dict(torch.load(best_model_path, map_location=device), strict=False)
    model = model.to(device).eval()

    # ✅ 타겟: layer 1, block 1
    layer_idx = 1
    block_idx = 1
    target_layer = model.global_branch.layers[layer_idx].blocks[block_idx].norm1
    reshape_fn = lambda x: reshape_transform_swin_by_layer(x, layer_idx)

    for batch_idx, batch in enumerate(test_loader):
    
        image, patches, labels, meta = batch
        image = image.to(device)
        patches = patches.to(device)
        labels = labels.to(device)
        # print(f"meta: {meta}")
        

        with torch.no_grad():
            logits = model(image, patches)
            probs = logits.squeeze(1)
            preds = (probs > 0.5).long()

        B = image.shape[0]
        for b in range(B):
            real = labels[b].item()
            pred = preds[b].item()
            patient_id = meta['patient_id'][b]
            # print(f"Patient ID: {patient_id}")

            if real != pred:
                continue  # ❌ 오분류 제외

            # ✅ 정답과 예측이 동일한 경우만 진행 (TP, TN)
            target_class = real
            class_name = dataset.get_class_name_from_label(real)
            file_prefix = f"{class_name}_real_{real}_pred_{pred}_batch_{batch_idx}_sample_{b}"

            out_dir = os.path.join(args.output_dir, binary_name, class_name)
            run_gradcam_single_block(
                image[b:b+1],
                patches[b:b+1],
                model,
                device,
                target_layer,
                reshape_transform_fn=reshape_fn,
                output_dir=out_dir,
                target_class=target_class,
                file_prefix=file_prefix,
                label = real,
                pred = pred,
                patient_id = patient_id,
                class_label_mapping = dataset.class_label_mapping
            )
        print(f"Batch {batch_idx} completed")

    print(f"[완료] Layer {layer_idx}, Block {block_idx} - 정답/예측 일치 CAM 저장 완료: {args.output_dir}")


if __name__ == '__main__':
    main()
