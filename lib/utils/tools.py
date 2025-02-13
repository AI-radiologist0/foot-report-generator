import torch
import os, re
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms

""" Check Device and Path for saving and loading """


def check_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')


def check_folder(log_dir):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    return log_dir


def find_latest_ckpt(folder):
    """ find latest checkpoint """
    files = []
    for fname in os.listdir(folder):
        s = re.findall(r'\d+', fname)
        if len(s) == 1:
            files.append((int(s[0]), fname))
    if files:
        file = max(files)[1]
        file_name = os.path.splitext(file)[0]
        previous_iter = int(file_name.split("_")[1])
        return file, previous_iter
    else:
        return None, 0


""" Training Tool for model """


def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag


def apply_gradients(loss, optim):
    optim.zero_grad()
    loss.backward()
    optim.step()


def infinite_iterator(loader):
    while True:
        for batch in loader:
            yield batch


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def cross_entroy_loss(logit, label):
    loss = torch.nn.CrossEntropyLoss()(logit, label)
    return loss


def accuracy(outputs, label):
    """ if you want to make custom accuracy for your model, you need to implement this function."""
    y = torch.argmax(outputs, dim=1)
    return (y.eq(label).sum())


def reduce_loss(tmp):
    """ will implement reduce_loss func """
    loss = tmp
    return loss


# def reduce_loss_dict(loss_dict):
#     world_size = get_world_size()
#
#     if world_size < 2:
#         return loss_dict
#
#     with torch.no_grad():
#         keys = []
#         losses = []
#
#         for k in sorted(loss_dict.keys()):
#             keys.append(k)
#             losses.append(loss_dict[k])
#
#         losses = torch.stack(losses, 0)
#         dist.reduce(losses, dst=0)
#
#         if dist.get_rank() == 0:
#             losses /= world_size
#
#         reduced_losses = {k: v.mean().item() for k, v in zip(keys, losses)}
#
#     return reduced_losses

""" Tool to set for model by loading config files """


def read_config(config_path):
    """ read config file """
    file = open(config_path, 'r')
    lines = file.read().split('\n')
    lines = [x for x in lines if x and not x.startswith('#')]
    lines = [x.rstrip().lstrip() for x in lines]

    return lines


def parse_model_config(config_path):
    """ Parse your model of configuration files and set module defines"""
    lines = read_config(config_path)
    module_configs = []

    for line in lines:
        if line.startswith('['):
            layer_name = line[1:-1].rstrip()
            if layer_name == "net":
                continue
            module_configs.append({})
            module_configs[-1]['type'] = layer_name

            if module_configs[-1]['type'] == 'convolutional':
                module_configs[-1]['batch_normalize'] = 0
        else:
            if layer_name == "net":
                continue
            key, value = line.split("=")
            value = value.strip()
            if value.startswith('['):
                module_configs[-1][key.rstrip()] = list(map(int, value[1:-1].rstrip().split(',')))
            else:
                module_configs[-1][key.rstrip()] = value.strip()

    return module_configs


""" If you want visualize_inference to fit your model, you need to implement below func."""


def visualize_inference(img, label, batch_size):
    """ Visualize Image Batch"""
    fig, axes = plt.subplots(1, batch_size, figsize=(10, 10))
    for i in range(batch_size):
        axes[i].imshow(np.squeeze(img[i]), cmap="gray")
        axes[i].set_title(f"predicted: {label[i]}")
        axes[i].axis('off')
    plt.show()


def visualize_feature_map(model, image):
    model.network.eval()

    # transformer
    # transform = mnist_transform()

    feature_map = None

    # input_tensor
    input_tensor = image.to(check_device())

    def hook(module, input, output):
        nonlocal feature_map
        feature_map = output.detach().cpu()

    target_layer = model.network.layers[6]
    hook_handle = target_layer.register_forward_hook(hook)

    with torch.no_grad():
        model.network(input_tensor)

    hook_handle.remove()

    plt.figure(figsize=(12, 8))
    for i in range(feature_map.size(1)):
        plt.subplot(4, 8, i + 1)
        plt.imshow(feature_map[0, i], cmap='viridis')
        plt.axis('off')
    plt.show()


def show_img(img):
    """ Display an img"""
    img = np.array(img, dtype=np.uint8)
    img = Image.fromarray(img)
    img.show()


def data_transform(img_size):
    transform_list = [
        transforms.Resize(size=[img_size, img_size]),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),  # [0, 255] -> [0, 1]
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),  # [0, 1] -> [-1, 1]
    ]
    return transforms.Compose(transform_list)


def mnist_transform():
    transforms_list = [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
    ]
    return transforms.Compose(transforms_list)
