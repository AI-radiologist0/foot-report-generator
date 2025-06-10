import _init_path

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import argparse
import seaborn as sns

import _init_path
from config import cfg, update_config
from dataset.joint_patches import FinalSamplesDataset
from utils.utils import stratified_split_dataset
from log_analysis import analyze_log_and_print_stats
from models.feature_extractor import get_feature_extractor
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import BinaryClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

def reshape_transform(tensor: torch.Tensor) -> torch.Tensor:
    """
    Swin-Transformer의 output shape이 [B, H, W, C]이면 [B, C, H, W]로 변환해줌.
    이미 [B, C, H, W]이면 그대로 반환.
    """
    print("reshape_transform input shape:", tensor.shape)
    
    if tensor.dim() != 4:
        raise ValueError(f"Expected 4D tensor but got shape {tensor.shape}")

    if tensor.shape[-1] == 768:  # [B, H, W, C]
        return tensor.permute(0, 3, 1, 2).contiguous()
    elif tensor.shape[1] == 768:  # [B, C, H, W]
        return tensor
    else:
        raise ValueError(f"Unexpected tensor shape: {tensor.shape}")



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

def run_gradcam(image, patches, model, device, output_dir, is_global=True, target_class=None, class_name=None, file_prefix=""):
    B = image.shape[0]
    if is_global:
        target_layer = (model.global_branch.layers[-1].blocks[-1].norm1 if model.global_is_swin
                        else model.global_branch.layer4)
    else:
        target_layer = (model.local_branch.layers[-1].blocks[-1].norm1 if model.local_is_swin
                        else model.local_branch.layer4)

    wrapper = ImagePatchWrapper(model, patches)
    wrapper.patches = patches

    # feature map 저장용 변수
    feature_map_storage = {}

    def hook_fn(module, input, output):
        feature_map_storage['feature_map'] = output

    handle = target_layer.register_forward_hook(hook_fn)

    if target_class is not None:
        targets = [BinaryClassifierOutputTarget(category=target_class)] * B
    else:
        raise ValueError("target_class must be specified.")

    cam = GradCAM(model=wrapper, 
                  target_layers=[target_layer],
                  reshape_transform=reshape_transform)
    grayscale_cam = cam(input_tensor=image, targets=targets)

    handle.remove()

    # Grad-CAM 결과 저장
    save_dir = os.path.join(output_dir, class_name)
    os.makedirs(save_dir, exist_ok=True)
    for b in range(B):
        img = image[b].detach().cpu().numpy().transpose(1, 2, 0)
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
        cam_img = show_cam_on_image(img, grayscale_cam[b], use_rgb=True)
        tag = "global" if is_global else "local"
        plt.imsave(os.path.join(save_dir, f'{file_prefix}gradcam_{tag}_b{b}_{class_name}.png'), cam_img)

def patchwise_local_branch_activation(model, patches, device):
    B, N, C, H, W = patches.shape
    activations = []
    for b in range(B):
        sample_acts = []
        for n in range(N):
            patch = patches[b, n].unsqueeze(0).to(device)
            in_channels = model.local_branch.conv1.in_channels
            if in_channels == 3:
                patch_input = patch
            else:
                patch_input = torch.zeros(1, in_channels, H, W, device=device)
                patch_input[:, :3, :, :] = patch
            with torch.no_grad():
                feat = model.local_branch(patch_input)
            act = feat.abs().mean().item()
            sample_acts.append(act)
        activations.append(sample_acts)
    return np.array(activations)

def save_patchwise_local_branch_activation(activations, output_dir, prefix="patch_local_branch_activation"):
    for b in range(activations.shape[0]):
        plt.figure(figsize=(12, 2))
        plt.bar(np.arange(activations.shape[1]), activations[b])
        plt.xlabel('Patch Index')
        plt.ylabel('Activation (local branch)')
        plt.title(f'Patch-wise Local Branch Activation (Sample {b})')
        plt.savefig(os.path.join(output_dir, f'{prefix}_b{b}.png'))
        plt.close()

def patchwise_local_branch_activation_from_feature_map(feature_map, num_patches=34):
    # feature_map: (B, C, H, W)
    B, C, H, W = feature_map.shape
    patch_channels = C // num_patches
    activations = []
    for b in range(B):
        acts = []
        for n in range(num_patches):
            patch_feat = feature_map[b, n*patch_channels:(n+1)*patch_channels, :, :]
            act = patch_feat.abs().mean().item()
            acts.append(act)
        activations.append(acts)
    return np.array(activations)

def save_patchwise_activation_heatmap(activations, output_dir, prefix="patch_local_branch_activation_heatmap"):
    # activations: (B, 34)
    for b in range(activations.shape[0]):
        plt.figure(figsize=(12, 1))
        sns.heatmap(activations[b][None, :], cmap='jet', cbar=True, xticklabels=np.arange(34), yticklabels=[])
        plt.xlabel('Patch Index')
        plt.title(f'Patch-wise Local Branch Activation (Sample {b})')
        plt.savefig(os.path.join(output_dir, f'{prefix}_b{b}.png'), bbox_inches='tight', pad_inches=0.1)
        plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_file', type=str, default='sampling_output/log_sampling_gout_normal_sw_res.log')
    parser.add_argument('--cfg', default="config/large/tmp/swin_t_resnet/origin_gout_normal_sampling20.yaml")
    parser.add_argument('--output_dir', type=str, default='patch_gradcam_output')
    parser.add_argument('--num_samples', type=int, default=2)
    args = parser.parse_args()

    update_config(cfg, args)
    os.makedirs(args.output_dir, exist_ok=True)

    best_model_path, best_seed = get_best_model_path(args.log_file, args.output_dir)
    print(f"Best model path: {best_model_path} (seed={best_seed})")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = FinalSamplesDataset(cfg)
    _, _, test_dataset = stratified_split_dataset(dataset, seed=best_seed)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.num_samples, shuffle=False)

    model = get_feature_extractor(cfg, is_train=False)
    model.load_state_dict(torch.load(best_model_path, map_location=device), strict=False)
    model = model.to(device).eval()

    class_label_mapping = dataset.class_label_mapping  # 예: {0: 'ra', 1: 'normal'}
    class_list = list(class_label_mapping.keys())

    for batch in test_loader:
        image, patches, labels, *rest = batch
        image = image.to(device)
        patches = patches.to(device)
        labels = labels.to(device)

        B = image.shape[0]

        for b in range(B):
            real_class = labels[b].item()
            real_class_name = dataset.get_class_name_from_label(real_class)

            # 각 클래스별로 CAM 저장
            for target_class in class_list:
                target_class_name = dataset.get_class_name_from_label(target_class)
                file_prefix = f"{real_class_name}_act_{target_class_name}_"

                save_dir = os.path.join(args.output_dir, target_class_name)
                os.makedirs(save_dir, exist_ok=True)

                run_gradcam(
                    image[b:b+1], patches[b:b+1], model, device,
                    output_dir=save_dir,
                    is_global=True,
                    target_class=target_class,
                    class_name=target_class_name,
                    file_prefix=file_prefix
                )
                # run_gradcam(
                #     image[b:b+1], patches[b:b+1], model, device,
                #     output_dir=save_dir,
                #     is_global=False,
                #     target_class=target_class,
                #     class_name=target_class_name,
                #     file_prefix=file_prefix
                # )
        break

    print(f"Grad-CAM 및 patch 활성화 결과가 {args.output_dir}에 저장되었습니다.")

if __name__ == '__main__':
    main()
