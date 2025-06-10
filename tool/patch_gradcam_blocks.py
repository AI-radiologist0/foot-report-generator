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
    """
    Swin Transformer의 layer별 출력 feature를 (B, C, H, W)로 변환
    """
    if tensor.dim() != 4:
        raise ValueError(f"Expected 4D tensor, got {tensor.shape}")

    B, H, W, C = tensor.shape
    size_map = {  # resolution by layer index
        0: (56, 56),
        1: (28, 28),
        2: (14, 14),
        3: (7, 7)
    }
    h, w = size_map.get(layer_idx, (7, 7))  # default 7x7

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


def run_gradcam_for_block(
    image, patches, model, device,
    output_root: str,
    layer_idx: int,
    block_idx: int,
    target_class: int = 0,
    file_prefix: str = ""
):
    B = image.shape[0]

    target_layer = model.global_branch.layers[layer_idx].blocks[block_idx].norm1
    reshape_fn = lambda x: reshape_transform_swin_by_layer(x, layer_idx)

    cam = GradCAM(
        model=ImagePatchWrapper(model, patches),
        target_layers=[target_layer],
        reshape_transform=reshape_fn
    )

    targets = [BinaryClassifierOutputTarget(category=target_class)] * B
    grayscale_cam = cam(input_tensor=image, targets=targets)

    save_dir = os.path.join(output_root, f"layer_{layer_idx}", f"block_{block_idx}")
    os.makedirs(save_dir, exist_ok=True)

    for b in range(B):
        img = image[b].detach().cpu().numpy().transpose(1, 2, 0)
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
        cam_img = show_cam_on_image(img, grayscale_cam[b], use_rgb=True)
        plt.imsave(os.path.join(save_dir, f"{file_prefix}gradcam_b{b}.png"), cam_img)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_file', type=str, default='sampling_output/log_sampling_gout_normal_sw_res.log')
    parser.add_argument('--cfg', type=str, default='config/large/tmp/swin_t_resnet/origin_gout_normal_sampling20.yaml')
    parser.add_argument('--output_dir', type=str, default='blockwise_gradcam_output')
    parser.add_argument('--num_samples', type=int, default=2)
    args = parser.parse_args()

    update_config(cfg, args)
    os.makedirs(args.output_dir, exist_ok=True)

    best_model_path, best_seed = get_best_model_path(args.log_file, args.output_dir)
    print(f"Best model path: {best_model_path} (seed={best_seed})")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = FinalSamplesDataset(cfg)
    _, _, test_dataset = stratified_split_dataset(dataset, seed=best_seed)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.num_samples, shuffle=False)

    model = get_feature_extractor(cfg, is_train=False)
    model.load_state_dict(torch.load(best_model_path, map_location=device), strict=False)
    model = model.to(device).eval()

    class_list = list(dataset.class_label_mapping.keys())

    for batch in test_loader:
        image, patches, labels, *rest = batch
        image = image.to(device)
        patches = patches.to(device)
        labels = labels.to(device)

        # ✅ 조건: normal(1), 질환(0) 둘 다 있을 때만 수행
        if not (0 in labels and 1 in labels):
            print("⏭️ 질환과 정상 둘 다 포함된 배치가 아님 — skip")
            continue

        B = image.shape[0]
        for b in range(B):
            real_class = labels[b].item()
            real_class_name = dataset.get_class_name_from_label(real_class)

            for target_class in class_list:
                target_class = real_class
                target_class_name = dataset.get_class_name_from_label(target_class)
                file_prefix = f"{real_class_name}_act_{target_class_name}_"
                output_root = os.path.join(args.output_dir, target_class_name)

                for layer_idx, layer in enumerate(model.global_branch.layers):
                    for block_idx in range(len(layer.blocks)):
                        run_gradcam_for_block(
                            image[b:b+1],
                            patches[b:b+1],
                            model=model,
                            device=device,
                            output_root=output_root,
                            layer_idx=layer_idx,
                            block_idx=block_idx,
                            target_class=target_class,
                            file_prefix=file_prefix
                        )
        break  # 첫 유효 배치만 수행

    print(f"모든 global layer/block에 대한 Grad-CAM 저장 완료: {args.output_dir}")


if __name__ == '__main__':
    main()
