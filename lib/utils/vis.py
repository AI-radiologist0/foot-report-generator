# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import numpy as np
import torchvision
import cv2

from core.inference import get_max_preds
from utils.transforms import flip_back

def save_batch_image_with_joints_fixed(batch_image, batch_joints, batch_joints_vis,
                                       file_name, nrow=8, padding=2, flip_pairs=None, is_flipped_list=None):
    '''
    batch_image: [batch_size, channel, height, width]
    batch_joints: [batch_size, num_joints, 3]
    batch_joints_vis: [batch_size, num_joints, 1]
    file_name: 저장할 파일명
    flip_pairs: keypoint 좌우 반전 매칭 리스트
    is_flipped_list: 각 이미지별 flip 여부 리스트 (배치 크기만큼 존재)
    '''
    if is_flipped_list is None:
        is_flipped_list = [False] * batch_image.size(0)  # 기본값: 모두 False

    grid = torchvision.utils.make_grid(batch_image, nrow, padding, True)
    ndarr = grid.mul(255).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
    ndarr = ndarr.copy()

    nmaps = batch_image.size(0)
    xmaps = min(nrow, nmaps)
    ymaps = int(math.ceil(float(nmaps) / xmaps))
    height = int(batch_image.size(2) + padding)
    width = int(batch_image.size(3) + padding)
    k = 0

    for y in range(ymaps):
        for x in range(xmaps):
            if k >= nmaps:
                break

            joints = batch_joints[k]
            joints_vis = batch_joints_vis[k]
            is_flipped = is_flipped_list[k]  # 🔥 개별적으로 flip 여부 확인

            # 🔥 Flip된 keypoint를 원래 위치로 복구
            if is_flipped and flip_pairs is not None:
                # X 좌표 반전
                joints[:, 0] = batch_image.size(3) - joints[:, 0] - 1  

                # Flip Pair 스왑
                for pair in flip_pairs:
                    joints[pair[0], :], joints[pair[1], :] = joints[pair[1], :], joints[pair[0], :]
                    joints_vis[pair[0], :], joints_vis[pair[1], :] = joints_vis[pair[1], :], joints_vis[pair[0], :]

            i = 0
            for joint, joint_vis in zip(joints, joints_vis):
                joint[0] = x * width + padding + joint[0]
                joint[1] = y * height + padding + joint[1]
                if joint_vis[0]:
                    cv2.circle(ndarr, (int(joint[0]), int(joint[1])), 2, [255, 0, 0], 1)
                    cv2.putText(ndarr, str(i), (int(joint[0]), int(joint[1])), 1, 1, [51, 255, 102], 1, cv2.LINE_AA)
                i += 1
            k += 1

    cv2.imwrite(file_name, ndarr)


def save_batch_heatmaps_fixed(batch_image, batch_heatmaps, file_name,
                              flip_pairs=None, is_flipped_list=None, normalize=True):
    '''
    batch_image: [batch_size, channel, height, width]
    batch_heatmaps: [batch_size, num_joints, height, width]
    file_name: 저장할 파일명
    flip_pairs: 좌우 keypoint 매칭 리스트
    is_flipped_list: 각 이미지별 flip 여부 리스트 (배치 크기만큼 존재)
    '''
    if is_flipped_list is None:
        is_flipped_list = [False] * batch_heatmaps.shape[0]  # 기본값: 모두 False

    if normalize:
        batch_image = batch_image.clone()
        min_val = float(batch_image.min())
        max_val = float(batch_image.max())
        batch_image.add_(-min_val).div_(max_val - min_val + 1e-5)

    batch_size = batch_heatmaps.shape[0]
    num_joints = batch_heatmaps.shape[1]
    heatmap_height = batch_heatmaps.shape[2]
    heatmap_width = batch_heatmaps.shape[3]

    grid_image = np.zeros((batch_size * heatmap_height,
                           (num_joints + 1) * heatmap_width,
                           3),
                          dtype=np.uint8)

    preds, maxvals = get_max_preds(batch_heatmaps)

    for i in range(batch_size):
        is_flipped = is_flipped_list[i]  # 🔥 개별적으로 flip 여부 확인

        heatmaps = batch_heatmaps[i]
        
        # 🔥 Flip된 heatmap을 원래대로 복구
        if is_flipped:
            heatmaps = flip_back(heatmaps[np.newaxis, :, :, :], flip_pairs)[0]

        image = batch_image[i].mul(255).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
        heatmaps = (heatmaps * 255).clip(0, 255).astype(np.uint8)

        resized_image = cv2.resize(image, (int(heatmap_width), int(heatmap_height)))

        height_begin = heatmap_height * i
        height_end = heatmap_height * (i + 1)

        for j in range(num_joints):
            cv2.circle(resized_image,
                       (int(preds[i][j][0]), int(preds[i][j][1])),
                       1, [0, 0, 255], 1)
            heatmap = heatmaps[j, :, :]
            colored_heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
            masked_image = colored_heatmap * 0.7 + resized_image * 0.3
            cv2.circle(masked_image,
                       (int(preds[i][j][0]), int(preds[i][j][1])),
                       1, [0, 0, 255], 1)

            width_begin = heatmap_width * (j + 1)
            width_end = heatmap_width * (j + 2)
            grid_image[height_begin:height_end, width_begin:width_end, :] = \
                masked_image

        grid_image[height_begin:height_end, 0:heatmap_width, :] = resized_image

    cv2.imwrite(file_name, grid_image)



def save_batch_image_with_joints(batch_image, batch_joints, batch_joints_vis,
                                 file_name, nrow=8, padding=2):
    '''
    batch_image: [batch_size, channel, height, width]
    batch_joints: [batch_size, num_joints, 3],
    batch_joints_vis: [batch_size, num_joints, 1],
    }
    '''
    grid = torchvision.utils.make_grid(batch_image, nrow, padding, True)
    ndarr = grid.mul(255).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
    ndarr = ndarr.copy()

    nmaps = batch_image.size(0)
    xmaps = min(nrow, nmaps)
    ymaps = int(math.ceil(float(nmaps) / xmaps))
    height = int(batch_image.size(2) + padding)
    width = int(batch_image.size(3) + padding)
    k = 0
    for y in range(ymaps):
        for x in range(xmaps):
            if k >= nmaps:
                break
            joints = batch_joints[k]
            joints_vis = batch_joints_vis[k]
            i = 0
            for joint, joint_vis in zip(joints, joints_vis):
                joint[0] = x * width + padding + joint[0]
                joint[1] = y * height + padding + joint[1]
                if joint_vis[0]:
                    cv2.circle(ndarr, (int(joint[0]), int(joint[1])), 2, [255, 0, 0], 1)
                    cv2.putText(ndarr, str(i), (int(joint[0]), int(joint[1])), 1, 1, [51, 255, 102], 1, cv2.LINE_AA)
                i += 1
            k = k + 1
    cv2.imwrite(file_name, ndarr)


def save_batch_heatmaps(batch_image, batch_heatmaps, file_name,
                        normalize=True):
    '''
    batch_image: [batch_size, channel, height, width]
    batch_heatmaps: ['batch_size, num_joints, height, width]
    file_name: saved file name
    '''
    if normalize:
        batch_image = batch_image.clone()
        min = float(batch_image.min())
        max = float(batch_image.max())

        batch_image.add_(-min).div_(max - min + 1e-5)

    batch_size = batch_heatmaps.size(0)
    num_joints = batch_heatmaps.size(1)
    heatmap_height = batch_heatmaps.size(2)
    heatmap_width = batch_heatmaps.size(3)

    grid_image = np.zeros((batch_size*heatmap_height,
                           (num_joints+1)*heatmap_width,
                           3),
                          dtype=np.uint8)

    preds, maxvals = get_max_preds(batch_heatmaps.detach().cpu().numpy())

    for i in range(batch_size):
        image = batch_image[i].mul(255)\
                              .clamp(0, 255)\
                              .byte()\
                              .permute(1, 2, 0)\
                              .cpu().numpy()
        heatmaps = batch_heatmaps[i].mul(255)\
                                    .clamp(0, 255)\
                                    .byte()\
                                    .cpu().numpy()

        resized_image = cv2.resize(image,
                                   (int(heatmap_width), int(heatmap_height)))

        height_begin = heatmap_height * i
        height_end = heatmap_height * (i + 1)
        for j in range(num_joints):
            cv2.circle(resized_image,
                       (int(preds[i][j][0]), int(preds[i][j][1])),
                       1, [0, 0, 255], 1)
            heatmap = heatmaps[j, :, :]
            colored_heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
            masked_image = colored_heatmap*0.7 + resized_image*0.3
            cv2.circle(masked_image,
                       (int(preds[i][j][0]), int(preds[i][j][1])),
                       1, [0, 0, 255], 1)

            width_begin = heatmap_width * (j+1)
            width_end = heatmap_width * (j+2)
            grid_image[height_begin:height_end, width_begin:width_end, :] = \
                masked_image
            # grid_image[height_begin:height_end, width_begin:width_end, :] = \
            #     colored_heatmap*0.7 + resized_image*0.3

        grid_image[height_begin:height_end, 0:heatmap_width, :] = resized_image

    cv2.imwrite(file_name, grid_image)





def save_debug_images(config, input, meta, target, joints_pred, output,
                      prefix):
    
    # is_flipped = meta['is_flipped']
    flip_pair = np.array(config.DATASET.FLIP_PAIR) - 1
    
    if not config.DEBUG.DEBUG:
        return

    if config.DEBUG.SAVE_BATCH_IMAGES_GT:
        save_batch_image_with_joints(
            input, meta['joints'], meta['joints_vis'],
            '{}_gt.jpg'.format(prefix)
        )
        # save_batch_image_with_joints_fixed(
        #     input, meta['joints'], meta['joints_vis'], '{}_gt+fixed.jpg'.format(prefix),
        #     flip_pairs=flip_pair, is_flipped_list=meta['is_flipped'] 
        # )
    if config.DEBUG.SAVE_BATCH_IMAGES_PRED:
        save_batch_image_with_joints(
            input, joints_pred, meta['joints_vis'],
            '{}_pred.jpg'.format(prefix)
        )
        # save_batch_image_with_joints_fixed(
        #     input, joints_pred, meta['joints_vis'], '{}_pred+fixed.jpg'.format(prefix),
        #     flip_pairs=flip_pair, is_flipped_list=meta['is_flipped'] 
        # )
 
    if config.DEBUG.SAVE_HEATMAPS_GT:
        save_batch_heatmaps(
            input, target, '{}_hm_gt.jpg'.format(prefix)
        )

    if config.DEBUG.SAVE_HEATMAPS_PRED:
        save_batch_heatmaps(
            input, output, '{}_hm_pred.jpg'.format(prefix)
        )

        
